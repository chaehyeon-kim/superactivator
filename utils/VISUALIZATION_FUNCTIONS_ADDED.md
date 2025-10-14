# Visualization Functions Added to text_visualization_utils.py

The following visualization functions were successfully extracted from text_visualization_utils_bak.py and added to text_visualization_utils.py:

## Main Visualization Functions:

1. **plot_sentence_similarity_heatmap** (lines 1680-1718)
   - Computes and visualizes pairwise cosine similarities between tokens in a sentence
   - Creates a heatmap showing token-to-token relationships

2. **plot_most_aligned_tokens** (lines 1719-1746) 
   - Plots the most aligned tokens for a concept using ChunkedActivationLoader
   - Shows top-k tokens with highest activations for a given concept

3. **plot_most_aligned_sentences** (lines 1747-1813)
   - Plots top-k most aligned sentences (CLS embeddings) for a concept
   - Supports filtering by train/test/cal splits

4. **plot_tokens_in_context_byconcept** (lines 1814-1876)
   - Visualizes tokens in context colored by concept activation
   - Shows paragraphs/sentences most/least activated by concepts

5. **plot_tokens_by_activation_and_gt** (lines 1877-2023)
   - Plots tokens showing paragraphs with most positive/negative/near-zero activations
   - Splits visualization by ground truth labels

## Helper Functions Added:

- **get_colormap_color** (new version at line 1621)
- **plot_colorbar** (new version at line 1625) 
- **highlight_tokens_with_legend** (new version at line 1649)
- **get_top_token_indices_for_concept** (new version at line 247)

## Notes:
- The file now contains both old and new versions of some helper functions
- The new visualization functions use the newer helper function versions
- All functions are fully compatible with ChunkedActivationLoader for efficient processing