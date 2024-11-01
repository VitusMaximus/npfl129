### Lecture: 5. MLP, Softmax as MaxEnt classifier, F1 score
#### Date: Oct 29
#### Slides: https://ufal.mff.cuni.cz/~courses/npfl129/2425/slides/?05
#### Reading: https://ufal.mff.cuni.cz/~courses/npfl129/2425/slides.pdf/npfl129-2425-05.pdf,PDF Slides
#### Questions: #lecture_5_questions

**Learning objectives.** After the lecture you shoud be able to

- Implement training of multi-layer perceptron using SGD.
- Explain the theoretical foundation behind the softmax activation function
  (including the necessary math).
- Choose a suitable evaluation metric for various classification tasks.

**Covered topics** and where to find more:

- Lagrange multipliers [Appendix E of PRML]
  - [A Simple Explanation of Why Lagrange Multipliers Works](https://medium.com/@andrew.chamberlain/a-simple-explanation-of-why-lagrange-multipliers-works-253e2cdcbf74), a blog post by Andrew Chamberlain
- Derivation of softmax via the maximum entropy principle [[The equivalence of logistic regression and maximum entropy models writeup](https://github.com/WinVector/Examples/blob/main/dfiles/LogisticRegressionMaxEnt.pdf)]
- $F_1$ score and $F_β$ score
- K-nearest neighbors [Section 2.5.2 of PRML]
