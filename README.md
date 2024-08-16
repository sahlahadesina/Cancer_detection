# Cancer_detection
This project focuses on the development of an intelligent cancer detection system utilizing machine learning techniques applied to medical images. Cancer remains a significant global health concern, with early detection playing a crucial role in patient outcomes. The objective is to create an assistive tool for pathologists, aiding in the identification of cancerous regions within biopsy microscopic images. The project involves designing and implementing machine learning algorithms to extract relevant features from image patches, subsequently employing a classifier to categorize these patches into either cancerous or non-cancerous classes.

The selected image set was sourced online from the MHIST dataset. This dataset comprises 3,152 images of colorectal polyps obtained from the Department of Pathology and Laboratory Medicine at Dartmouth-Hitchcock Medical Center. Each image has been categorized as either HP or SSA by seven pathologists. HPs are generally benign, whereas SSAs are precancerous lesions with the potential to develop into cancer. 
![image](https://github.com/user-attachments/assets/a79d7b40-6c8a-4947-b60b-2467d8b8f0e7)

To train the model, I opted for transfer learning using the ResNet101 architecture. 
Once training was completed, the model's performance was evaluated by assessing the train accuracy, test accuracy, classification report, and confusion matrix. The maximum Accuracy I was able to achieve was 98.6. 
