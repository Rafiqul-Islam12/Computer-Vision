# ***Convolutional Neural Networks (CNN)***

---
<img src="https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/Convolutional%20Neural%20Networks%20(CNN)/images/img01.png" width="900">  

---
- ***CNN (Convolutional Neural Network) প্রথম Yann LeCun ১৯৮০-এর দশকে (1980s) introduce করেন।***  
- ***শুরুতে এটি handwritten digit recognition-এর জন্য তৈরি হয়েছিল।***  
- ***মূলত postal sector-এ ZIP code ও PIN code পড়তে ব্যবহার করা হতো।***   
- ***2012 সালে Alex Krizhevsky, ImageNet dataset এবং GPU ব্যবহারের মাধ্যমে CNN আবার জনপ্রিয় হয়ে ওঠে।***

---
- ***CNN (Convolutional Neural Network) হলো একটি Deep Learning model, যা মূলত Computer Vision-এর কাজে ব্যবহার করা হয়।***    
- ***CNN-কে ConvNet নামেও ডাকা হয়।***  
- ***এটি একটি Feed-Forward Neural Network, যেখানে data শুধুমাত্র Input Layer → Hidden Layer → Output Layer-এর দিকে প্রবাহিত হয়; কোনো feedback loop থাকে না।***  
- ***CNN একটি image-কে input হিসেবে গ্রহণ করে এবং বিভিন্ন filter (kernel) ব্যবহার করে image-এর গুরুত্বপূর্ণ features যেমন edge, corner, texture, shape ইত্যাদি automatically learn করে।***  
- ***Traditional Machine Learning-এ feature manually extract করতে হতো, কিন্তু CNN নিজেই (automatically) feature শিখে নেয় এবং সেই feature ব্যবহার করে final classification বা prediction করে।***

---
***CNN consists of 3 Layers. Those are***   
***1. Convolution Layer***   
***2. Pooling Layer***   
***3. Fully Connected Layer or Dense Layer***   

---
***CNN একটি image-কে input হিসেবে গ্রহণ করে এবং সেটিকে dog, cat, lion, tiger ইত্যাদি category-তে classify করে।***  
***Computer একটি image-কে pixels-এর array (matrix) হিসেবে দেখে।***  
***Image-এর size বা resolution অনুযায়ী image-কে h × w × d আকারে represent করা হয়।***   
- ***h = Height***  
- ***w = Width***    
- ******d = Depth (Channel/Dimension)***

***RGB image-এর 3টি channel (Red, Green, Blue) থাকে, তাই এর dimension 3। যেমন: 6 × 6 × 3।***  
***Grayscale image-এ মাত্র 1টি channel থাকে, তাই এর dimension 1। যেমন: 4 × 4 × 1।***   
