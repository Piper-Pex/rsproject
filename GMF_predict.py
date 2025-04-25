

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
import h5py
# ----------------------
# Load metadata 
# ----------------------
def load_metadata(filename):
    """Metadata loading function identical to the training code"""
    with h5py.File(filename, "r") as f:
        songs_dataset = f['metadata/songs']  # Note the hierarchy uses '/' separators
        
        # Extract all required fields
        song_ids_bytes = songs_dataset['song_id'][()]
        titles_bytes = songs_dataset['title'][()]
        artists_bytes = songs_dataset['artist_name'][()]  # Added artist field
        
        # Unified decoding process
        decode_func = lambda x: x.decode('utf-8', errors='ignore').strip()
        song_ids = list(map(decode_func, song_ids_bytes))
        titles = list(map(decode_func, titles_bytes))
        artists = list(map(decode_func, artists_bytes))
        
        # Build DataFrame
        df = pd.DataFrame({
            'song_id': song_ids,
            'title': titles,
            'artist_name': artists  # Add other required fields
        })
        
        # Filter invalid data
        return df[df['song_id'].str.len() > 0]

# ----------------------
# Enhanced Recommendation System Class
# ----------------------
class AdvancedMusicRecommender:
    def __init__(self):
        # Load metadata
        self.metadata = load_metadata('msd_summary_file.h5')
        
        # Load model and encoders
        self.model = tf.keras.models.load_model('best_gmf_model.keras')
        self.song_encoder = joblib.load('song_encoder_gmf.pkl')
        
        # Handle missing max_play
        try:
            self.max_play = joblib.load('max_play_gmf.pkl')
        except FileNotFoundError:
            print("Warning: Estimating max_play using metadata")
            self.max_play = self.metadata['plays'].max() if 'plays' in self.metadata else 1
        
        # Create song ID to index mapping
        self.song_id_to_idx = {
            song_id: idx 
            for idx, song_id in enumerate(self.song_encoder.classes_)
        }
        
        # Get song embeddings
        self.song_embeddings = self.model.get_layer('item_embed').get_weights()[0]

    def search_songs(self, query, top_k=5):
        """Modified search function with consistent fields"""
        mask = (
            self.metadata['title'].str.contains(query, case=False) |
            self.metadata['artist_name'].str.contains(query, case=False)
        )
        return self.metadata[mask].head(top_k)[['song_id', 'title', 'artist_name']]
    
    def create_virtual_user(self, song_ids):
        """Create virtual user features from song IDs"""
        valid_ids = [song_id for song_id in song_ids if song_id in self.song_id_to_idx]
        
        if not valid_ids:
            raise ValueError("No valid song IDs found")
            
        indices = [self.song_id_to_idx[song_id] for song_id in valid_ids]
        avg_embedding = np.mean(self.song_embeddings[indices], axis=0)
        return avg_embedding

    def _select_songs_interactively(self, matched_songs):
        """Interactive song selection with re-search option"""
        print("\n🔍 Found matching songs:")
        print("0. Search again (unsatisfied with results)")
        for idx, (_, row) in enumerate(matched_songs.iterrows(), 1):
            print(f"{idx}. {row['title']} - {row['artist_name']}")
        
        while True:
            try:
                selected = input("Enter song numbers (space-separated, 0 to re-search, enter for all): ").strip()
                if not selected:
                    return matched_songs['song_id'].tolist()
                
                if '0' in selected.split():
                    return None
                
                indices = list(map(int, selected.split()))
                valid_indices = [i for i in indices if 1 <= i <= len(matched_songs)]
                
                if not valid_indices:
                    print("⚠️ Invalid input, please try again")
                    continue
                
                return matched_songs.iloc[[i-1 for i in valid_indices]]['song_id'].tolist()
            
            except ValueError:
                print("⚠️ Please enter valid numbers")

    def _format_grouped_results(self, grouped_results):
        """Format grouped search results with hierarchical numbering"""
        formatted = []
        for group_idx, (query, results) in enumerate(grouped_results, 1):
            if not results.empty:
                formatted.append(f"\n🔍 Results for: '{query}'")
                for item_idx, (_, row) in enumerate(results.iterrows(), 1):
                    formatted.append(f"{group_idx}.{item_idx} {row['title']} - {row['artist_name']}")
            else:
                formatted.append(f"\n⚠️ No results found for: '{query}'")
        return "\n".join(formatted)

    def _parse_group_selection(self, selection, grouped_results):
        """Parse hierarchical selection like '1.1 2.3'"""
        selected_ids = []
        valid_groups = [g for g in grouped_results if not g[1].empty]
        
        for part in selection.split():
            try:
                group_num, item_num = map(int, part.split('.'))
                # Adjust for valid groups only
                if 1 <= group_num <= len(valid_groups):
                    group_query, group_df = valid_groups[group_num-1]
                    if 1 <= item_num <= len(group_df):
                        selected_ids.append(group_df.iloc[item_num-1]['song_id'])
            except:
                continue
        return selected_ids
    
    def generate_recommendations(self, input_titles, top_n=10, verbose=True):
        """
        Core recommendation generation function (Fixed Version)
        """
        try:
            # Step 1: Process each query separately
            grouped_results = []
            valid_queries = 0

            for query in input_titles:
                query = query.strip()
                if not query:
                    continue

                results = self.search_songs(query)
                grouped_results.append((query, results))
                if not results.empty:
                    valid_queries += 1

            # Step 2: Display grouped results
            if verbose:
                print("\n" + "="*50)
                print(self._format_grouped_results(grouped_results))
                print("="*50 + "\n")

            # Step 3: Interactive selection
            selected_ids = []
            while True:
                try:
                    selection = input(
                        "Enter selections (e.g. '1.1 2.3'), '0' to re-search, or enter to confirm: "
                    ).strip()
                    
                    if selection == '0':
                        return None
                    if not selection:
                        break
                        
                    selected_ids = self._parse_group_selection(selection, grouped_results)
                    if not selected_ids:
                        print("⚠️ No valid selections, try again")
                        continue
                    break
                        
                except KeyboardInterrupt:
                    print("\n⏹ Selection canceled")
                    if input("Continue? (y/n): ").lower() == 'n':
                        return pd.DataFrame()

            # Step 4: Create virtual user
            try:
                if verbose:
                    print("\n⭐ Analyzing song features...")
                
                virtual_user = self.create_virtual_user(selected_ids)
            except ValueError as e:
                if verbose:
                    print(f"❌ Feature analysis failed: {str(e)}")
                return pd.DataFrame()
            except Exception as e:
                if verbose:
                    print(f"❌ Unexpected error: {str(e)}")
                return pd.DataFrame()

            # Step 5: Calculate similarity (Fixed)
            try:
                if verbose:
                    print("🔢 Calculating similarities...")
                
                scores = np.dot(self.song_embeddings, virtual_user)
                
                # Use the current selected ID to exclude the selected songs
                input_indices = [
                    self.song_id_to_idx[sid] 
                    for sid in selected_ids  # use the actual selected ID instead of all_matches
                    if sid in self.song_id_to_idx
                ]
                scores[input_indices] = -np.inf  # discard the selected songs
            except Exception as e:
                if verbose:
                    print(f"❌ Similarity calculation failed: {str(e)}")
                return pd.DataFrame()

            # Step 6: Generate recommendations
            try:
                top_indices = np.argsort(scores)[-top_n:][::-1]
                top_scores = scores[top_indices]
                top_song_ids = self.song_encoder.inverse_transform(top_indices)

                recommendations = self.metadata[
                    self.metadata['song_id'].isin(top_song_ids)
                ].copy()
                
                try:
                    recommendations['predicted_plays'] = np.clip(
                        top_scores * self.max_play, 
                        0,
                        None
                    )
                except:
                    recommendations['predicted_plays'] = 0

                return recommendations.sort_values('predicted_plays', ascending=False)

            except Exception as e:
                if verbose:
                    print(f"❌ Recommendation generation failed: {str(e)}")
                return pd.DataFrame()

        except KeyboardInterrupt:
            print("\n⏹ Recommendation process interrupted")
            return pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"❗ Unhandled exception: {str(e)}")
            return pd.DataFrame()

# ----------------------
# Interactive Recommendation Flow
# ----------------------
def interactive_recommendation():
    recommender = AdvancedMusicRecommender()
    
    while True:
        print("\n🎵 Please enter your favorite songs/artists (enter 'exit' to quit):")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        queries = [q.strip() for q in user_input.split(',')]
        
        while True:
            result = recommender.generate_recommendations(queries)
            if result is None:
                break
            elif not result.empty:
                print("\n🎧 Recommended songs for you:")
                print(result[['title', 'artist_name', 'predicted_plays']]
                    .head(10).to_string(index=False))
                break
            else:
                print("Unable to generate recommendations, please try different input")
                break

if __name__ == "__main__":
    interactive_recommendation()