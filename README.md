# Image text retrieval using joint embedding space
### CS679: Neural Networks final project


Combining visual and textual domains into a common space has been an exciting research domain. In this work,
a method is implemented to learn image-text joint embeddings,
where two classification models, separate for each domain, are
combined to form a common representation. These classification
models are further combined and trained using multi-component
loss functions, where cross-view and within-view constraints
are added to optimize Bi-directional ranking and to preserve
structure between matching pairs. The margin values for such constraints and weight multipliers of individual
loss components were experimented with. The results reveal that faster training times can
be achieved for such algorithms by using variable margins and weight multipliers.
