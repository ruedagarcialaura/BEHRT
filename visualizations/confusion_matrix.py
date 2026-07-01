import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Calculate the confusion matrix using your Colab variables
cm = confusion_matrix(all_labels, all_preds)

# 2. Set up the plot
plt.figure(figsize=(8, 6))

# 3. Create a beautiful heatmap
# cmap='Blues' gives it a professional blue clinical tone
# fmt='d' ensures numbers are integers, not scientific notation
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Healthy (0)', 'Predicted Diabetic (1)'], 
            yticklabels=['True Healthy (0)', 'True Diabetic (1)'],
            annot_kws={"size": 14}) # Make numbers larger

# 4. Add titles and labels
plt.title('Confusion Matrix - Type 2 Diabetes Prediction', fontsize=16, pad=15)
plt.xlabel('Model Prediction', fontsize=14, labelpad=10)
plt.ylabel('Actual Ground Truth', fontsize=14, labelpad=10)

# 5. Fix layout and save the image
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # dpi=300 for high quality in your thesis
plt.show()

print("Confusion matrix successfully generated and saved as 'confusion_matrix.png'")