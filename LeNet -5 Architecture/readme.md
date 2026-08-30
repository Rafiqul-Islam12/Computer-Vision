# ***LeNet-5 Architecture***

***LeNet-5 হলো একটা classic Convolutional Neural Network (CNN) architecture, যেটা 1998 সালে Yann LeCun design করেছিলেন — মূলত handwritten digit recognition (MNIST dataset) এর জন্য। এটাকে CNN এর "grandfather" বলা হয়, কারণ modern CNN গুলোর (AlexNet, VGG, ResNet) basic idea এখান থেকেই এসেছে।***  

<img src='https://github.com/Rafiqul-Islam12/Computer-Vision/blob/main/LeNet%20-5%20Architecture/images/img.png' width=800>   

# ***Model Code***

```python
model = Sequential()

model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='tanh', input_shape=(32,32,1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))
```

# ***Layer by Layer Explanation***

## 🔹 Input Layer
- **Shape**: 32×32×1
- Grayscale image, single channel।
- MNIST images আসলে 28×28, কিন্তু padding দিয়ে 32×32 করা হয় যাতে edge features ভালোভাবে capture হয়।
    
## 🔹 ***Layer 1 (C1): Convolutional Layer***
```python
Conv2D(6, kernel_size=(5,5), padding='valid', activation='tanh', input_shape=(32,32,1))
```
- **Filters**: 6টা, প্রতিটার size 5×5
- **Padding**: valid (কোনো padding add হয় না)
- **Calculation**: (32-5)/1 + 1 = 28
- **Output**: 28×28×6
- **কাজ**: raw pixel থেকে basic features যেমন edges, lines, curves বের করে
- **Parameters**: (5×5×1+1)×6 = 156টা

## 🔹 Layer 2 (S2): Average Pooling Layer
```python
AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')
```
- **Calculation**: 28/2 = 14
- **Output**: 14×14×6
- **কাজ**:
  - feature map এর size অর্ধেক করে দেয় (downsampling)
  - computation কমায়
  - image এ সামান্য shift/rotation হলেও model robust থাকে (translation invariance)
- **Parameters**: 0 (কোনো learnable parameter নেই, শুধু averaging হয়)

## 🔹 Layer 3 (C3): Convolutional Layer
```python
Conv2D(16, kernel_size=(5,5), padding='valid', activation='tanh')
```
- **Filters**: 16টা, size 5×5
- **Calculation**: (14-5)/1 + 1 = 10
- **Output**: 10×10×16
- **কাজ**: আগের layer এর simple features গুলো combine করে আরও complex/abstract features (shapes, patterns) শেখে
- **Parameters**: (5×5×6+1)×16 = 2,416টা

## 🔹 Layer 4 (S4): Average Pooling Layer
```python
AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')
```
- **Calculation**: 10/2 = 5
- **Output**: 5×5×16
- **কাজ**: আবার downsampling, feature map ছোট করা
- **Parameters**: 0

## 🔹 Flatten Layer
```python
Flatten()
```
- **Calculation**: 5×5×16 = 400
- **Output**: 400 (1D vector)
- **কাজ**: 3D feature map কে 1D vector এ convert করে, যাতে Dense layer এ পাঠানো যায়
- **Parameters**: 0

## 🔹 ***Layer 5 (C5/F5): Fully Connected Layer***
```python
Dense(120, activation='tanh')
```
- **Output**: 120
- **কাজ**: সব extracted features গুলোকে একসাথে combine করে high-level representation তৈরি করে
- **Parameters**: (400×120)+120 = 48,120টা

## 🔹 ***Layer 6 (F6): Fully Connected Layer***
```python
Dense(84, activation='tanh')
```
- **Output**: 84
- **কাজ**: features আরও refine/condense করে, final classification এর জন্য প্রস্তুত করে
- **Parameters**: (120×84)+84 = 10,164টা

## 🔹 ***Output Layer***
```python
Dense(10, activation='softmax')
```
- **Output**: 10 (digit 0-9 এর জন্য 10টা class)
- **কাজ**: প্রতিটা class এর probability বের করে, সর্বোচ্চ probability যেটার, সেটাই predicted digit
- **Parameters**: (84×10)+10 = 850টা

# 📊 ***Complete Summary Table***

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Input | - | 32×32×1 | 0 |
| C1 | Conv2D | 28×28×6 | 156 |
| S2 | AvgPool | 14×14×6 | 0 |
| C3 | Conv2D | 10×10×16 | 2,416 |
| S4 | AvgPool | 5×5×16 | 0 |
| Flatten | - | 400 | 0 |
| C5/F5 | Dense | 120 | 48,120 |
| F6 | Dense | 84 | 10,164 |
| Output | Dense | 10 | 850 |

**Total Trainable Parameters** ≈ 61,706টা

# ***Reference***
***Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *"Gradient-Based Learning Applied to Document Recognition."* Proceedings of the IEEE, 1998.***

