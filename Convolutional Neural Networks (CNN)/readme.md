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
- ***d = Depth (Channel/Dimension)***

***RGB image-এর 3টি channel (Red, Green, Blue) থাকে, তাই এর dimension 3। যেমন: 6 × 6 × 3।***  
***Grayscale image-এ মাত্র 1টি channel থাকে, তাই এর dimension 1। যেমন: 4 × 4 × 1।***   

---
# ***1. CONVOLUTION LAYER (CONVOLUTION OPERATION)***    
- ***এটি input image থেকে important features extract করে।***   
- ***এটি edges, lines, texture, shapes ইত্যাদি pattern শিখে এবং image-এর যেকোনো স্থানে সেই pattern চিনতে পারে।***   
- ***শুরুতে horizontal ও vertical edges detect করে, পরে এগুলো ব্যবহার করে high-level features তৈরি করে।***   
   
***Convolution Operation-এ ৩টি জিনিস থাকে:***  
- ***Input Image***  
- ***Filter/Kernel (Feature Detector)***  
- ***Feature Map (Output)***
    
---
- ***একটি ছোট filter (যেমন 3×3) input image-এর উপর slide করে।***   
- ***প্রতিটি position-এ element-wise multiplication এবং summation করা হয়।***   
- ***এর ফলে একটি Feature Map তৈরি হয়, যা পরবর্তী layer-এ পাঠানো হয়।***   
<img src="https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/Convolutional%20Neural%20Networks%20(CNN)/images/img02.gif" width="400">   

- ***Convolution হলো element-wise multiplication + summation।***  
- ***Filter পুরো image scan করে feature বের করে।***
- ***Convolution-এর output-কে Feature Map বলা হয়।***
- ***সাধারণত Convolution-এর পরে image-এর size ছোট হয়ে যায় (padding না থাকলে)।***

---
## ***Filter (Kernel)***  
***Filter একটি ছোট matrix, যা নির্দিষ্ট feature detect করে।***  
***বিভিন্ন filter বিভিন্ন feature detect করতে পারে, যেমন:***  
- ***Vertical Edge***  
- ***Horizontal Edge***  
- ***Diagonal Edge***  
  
***Edge Detection Filter → Image-এর edges বের করে।***    
***Blur Filter → Image smooth বা ঝাপসা করে।***  
***Sharpen Filter → Image আরও clear ও sharp করে।***  
***Vertical Filter → Vertical edges detect করে।***  
***Horizontal Filter → Horizontal edges detect করে।***  

<img src="https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/Convolutional%20Neural%20Networks%20(CNN)/images/img03.png" width="600">  
<img src="https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/Convolutional%20Neural%20Networks%20(CNN)/images/img04.png" width="600">  

---
## ***Convolution Operation Example***  
***This is a basic example with a 2 × 2 kernel:***   
<img src="https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/Convolutional%20Neural%20Networks%20(CNN)/images/img05.png" width="500">  
- ***(0 × 0) + (1 × 1) + (3 × 2) + (4 × 3) = 19***  
- ***(1 × 0) + (2 × 1) + (4 × 2) + (5 × 3 ) = 25***  
- ***(3 × 0) + (4 × 1) + (6 × 2) + (7 × 3) = 37***  
- ***(4 × 0) + (5 × 1) + (7 × 2) + (8 × 3) = 43***
   
***The output matrix of this process is known as the Feature map.***  

---






