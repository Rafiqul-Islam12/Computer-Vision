# Transfer Learning দিয়ে Bone Fracture Classification

এই project এ **Transfer Learning** ব্যবহার করে একটা Deep Learning model বানানো হয়েছে, যেটা একটা bone X-ray image দেখে বলে দিতে পারে সেটা **Oblique Fracture** নাকি **Spiral Fracture**। এই README টাতে concept + code, দুটোই একসাথে আছে, যাতে পরে দরকার হলে সরাসরি code copy করে ব্যবহার করা যায়, আবার concept ও clear থাকে।

---

## 🧠 প্রথমে বুঝি — Transfer Learning জিনিসটা আসলে কী?

ধরো, তুমি ইতিমধ্যে **হাজার হাজার প্রাণী, গাড়ি, ফল, মানুষের মুখ** চিনতে শিখে ফেলা একজন expert কে চেনো। এখন তুমি চাও ও তোমাকে **bone fracture** চিনতে শেখাবে। তুমি কি ওকে আবার শুরু থেকে "এটা কী একটা line, এটা কী একটা curve" — এসব শেখাবে?

না! কারণ ও already জানে edges, shapes, textures, patterns কীভাবে চিনতে হয়। তুমি শুধু ওকে **নতুন একটা specific task** এ focus করাবে — bone fracture এর দুটো type চেনা।

এটাই **Transfer Learning** এর মূল idea:

> একটা model কে বিশাল dataset এ (যেমন ImageNet — 1.2 মিলিয়ন image, 1000 category) আগে থেকেই train করা থাকে। সেই "শেখা জ্ঞান" (learned features) আমরা reuse করি আমাদের নিজের ছোট, specific problem এর জন্য — পুরো model কে শুরু থেকে train না করেই।

এই approach এ দুইটা বড় সুবিধা পাওয়া যায়:
- **কম data লাগে** (আমাদের কাছে হয়তো মাত্র কয়েকশো X-ray image আছে, কোটি কোটি না)
- **কম সময় আর কম compute power লাগে** (পুরো network নতুন করে train করার দরকার নেই)

---

## 🏗️ Step 1: Environment আর GPU প্রস্তুত করা

Deep Learning model, বিশেষ করে VGG16 এর মতো বড় network train করতে অনেক computation লাগে, তাই **GPU** ব্যবহার করা হয়। Google Colab এ কোন GPU assign হয়েছে সেটা এভাবে চেক করা যায়:

```python
!nvidia-smi
```

এখানে একটা **Tesla T4 GPU** পাওয়া গেছে, যেটা training কে অনেক দ্রুত করবে (CPU দিয়ে করলে এই কাজ ঘণ্টার পর ঘণ্টা লাগতো)।

---

## 📁 Step 2: Dataset Extract আর Organize করা

আমাদের dataset একটা zip file আকারে ছিল, সেটা extract করা হয়েছে:

```python
!unzip /content/Bones-data.zip
```

এখানে একটা গুরুত্বপূর্ণ জিনিস লক্ষ্য করার মতো — dataset টা **folder structure দিয়ে labeled**:

```
Bones-data/
├── train/
│   ├── Oblique fracture/   → এই folder এর সব image "Oblique fracture" class এর
│   └── Spiral Fracture/    → এই folder এর সব image "Spiral Fracture" class এর
└── test/
    ├── Oblique fracture/
    └── Spiral Fracture/
```

এটা Deep Learning এ খুবই common একটা convention — তোমাকে আলাদা করে label file বানাতে হয় না, শুধু image গুলো তাদের নিজের class-নামের folder এ রাখলেই হয়। পরে Keras এর একটা tool (`flow_from_directory`) automatically folder name গুলো কে label হিসেবে ধরে নেবে।

Train আর test path define করা হয়েছে এভাবে:

```python
train_path = '/content/Bones-data/train'
valid_path = '/content/Bones-data/test'
```

আমরা `train` আর `test` (validation) — এই দুই ভাগে data ভাগ করে রাখলাম যাতে model কে যেই data দিয়ে train করাচ্ছি, সেটা দিয়েই performance check না করে, **আগে না দেখা** data দিয়ে সততার সাথে যাচাই করতে পারি।

---

## 📐 Step 3: Image Size ঠিক করা — কেন ঠিক 224×224?

```python
IMAGE_SIZE = [224, 224]
```

```python
size = [224, 224] + [3]
size
```

সব image কে **224×224 pixel** size এ নিয়ে আসা হয়েছে, সাথে 3টা color channel (Red, Green, Blue) — অর্থাৎ প্রতিটা image এর shape হবে `(224, 224, 3)`।

এই সংখ্যাটা random না। যেই pretrained model (VGG16) আমরা ব্যবহার করবো, সেটা originally ঠিক এই size এর input দিয়েই train হয়েছিল ImageNet dataset এ। তাই আমাদের নতুন data ও এই একই size এ আনতে হবে, নাহলে pretrained network এর সাথে match হবে না।

---

## 📦 Step 4: প্রয়োজনীয় Library গুলো Import করা

```python
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
```

- `VGG16` → pretrained model, যেটাই transfer learning এর মূল অংশ
- `ImageDataGenerator` → image augmentation আর loading এর জন্য
- `glob` → folder এর ভিতর file/folder list করার জন্য
- `Sequential, Dense, Flatten` → নতুন custom layers বানানোর জন্য

---

## 🎯 Step 5: VGG16 — আমাদের "Expert" কে নিয়ে আসা

```python
vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
```

VGG16 হলো একটা বিখ্যাত CNN architecture, যেটা **ImageNet** dataset এ train করা হয়েছিল — যেখানে 1.2 মিলিয়ন image আর 1000 রকম category আছে। এই training process এ VGG16 শিখে ফেলেছে কীভাবে:

- Edge আর line চিনতে হয় (shallow layers এ)
- Texture আর pattern চিনতে হয় (middle layers এ)
- Complex shape আর object চিনতে হয় (deep layers এ)

এই "চেনার ক্ষমতা"টাই আমরা borrow করবো। কিন্তু একটা জিনিস বাদ দিতে হবে — VGG16 এর একদম শেষের অংশ, যেটা 1000টা ImageNet class predict করে। কারণ আমাদের দরকার মাত্র 2টা class (Oblique আর Spiral fracture)। তাই VGG16 কে load করার সময় বলে দেওয়া হয় "top" (শেষের classification layer) বাদ দিয়ে শুধু middle অংশটা (feature extractor অংশ) দাও — এটাকেই বলে **`include_top=False`**।

তাহলে VGG16 এখন কাজ করছে একটা **powerful feature extractor** হিসেবে — একটা image দিলে, ও সেটার ভেতরের meaningful patterns বের করে দেবে, কিন্তু final decision নিজে নেবে না।

---

## ❄️ Step 6: Layer গুলো "Freeze" করা — কেন এটা এত গুরুত্বপূর্ণ?

প্রথমে দেখে নেওয়া যাক প্রতিটা layer trainable কিনা (default `True` থাকে):

```python
for layer in vgg16.layers:
  print(layer.trainable)
```

এখন VGG16 এর সব layer **freeze** করে দেওয়া হচ্ছে:

```python
for layer in vgg16.layers:
    layer.trainable = False
```

আর verify করা হচ্ছে সব layer এখন frozen কিনা:

```python
for layer in vgg16.layers:
  print(layer.name, layer.trainable)
```

VGG16 ইতিমধ্যে লক্ষ লক্ষ image দেখে ভালো feature চেনা শিখে ফেলেছে। যদি আমরা ওকে আমাদের অল্প data (মাত্র কয়েকশো X-ray) দিয়ে আবার train হতে দিই, তাহলে দুটো সমস্যা হতে পারে:
1. এত অল্প data দিয়ে এত বড় network train করলে **overfitting** হয়ে যাবে (model শুধু training data মুখস্থ করে ফেলবে, নতুন data তে fail করবে)
2. আগে যেই ভালো, general feature গুলো শেখা ছিল, সেগুলো নষ্ট হয়ে যেতে পারে (একে বলে **catastrophic forgetting**)

তাই আমরা `layer.trainable = False` করে বলে দিই, "এই layer গুলোর weights training এর সময় update হবে না, এরা যা জানে তাই থাকবে।" শুধু আমরা যেই নতুন layer গুলো উপরে যোগ করবো, সেগুলোই training এর সময় শিখবে। এটাকেই বলে **Feature Extraction based Transfer Learning**।

VGG16 এর পুরো architecture দেখতে:

```python
vgg16.summary()
```

---

## 🧩 Step 7: নতুন Layer যোগ করা — আমাদের নিজের "Decision Maker" বানানো

প্রথমে dataset এর class (folder) গুলো বের করা হচ্ছে:

```python
folders = glob('/content/Bones-data/train/*')
folders
```

```python
num_of_class = len(folders)
num_of_class
```

VGG16 তো শুধু features বের করবে, কিন্তু "এটা Oblique নাকি Spiral" — এই সিদ্ধান্তটা কে নেবে? এর জন্য VGG16 এর উপরে নতুন layer বসানো হচ্ছে:

```python
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_of_class, activation='softmax'))
```

- **Flatten Layer**: VGG16 থেকে যেই feature maps বের হয়, সেগুলো multi-dimensional হয়। Flatten layer সেগুলোকে একটা লম্বা 1D vector এ রূপান্তর করে।
- **Dense Layer (256 neurons, ReLU)**: এটা একটা নতুন "শেখার layer", যেটা VGG16 এর বের করা features গুলো নিয়ে আমাদের specific problem এর জন্য useful combination তৈরি করবে। এই layer টাই মূলত train হবে।
- **Dense Layer (output, Softmax)**: শেষ layer, যেটা `num_of_class` (এখানে 2টা) class এর probability বের করে দেয়।

পুরো model টা দাঁড়ালো এরকম:

```
[Input Image] → [VGG16 (Frozen — শুধু feature বের করে)] → [Flatten] → [Dense 256 (নতুন, trainable)] → [Dense 2, Softmax (নতুন, trainable)] → [Output: Oblique / Spiral]
```

Model এর structure দেখতে:

```python
model.summary()
```

---

## ⚙️ Step 8: Model কে "Compile" করা — Training এর নিয়ম ঠিক করা

```python
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
```

- **Loss Function (`categorical_crossentropy`)**: model এর prediction কতটা ভুল হচ্ছে সেটা measure করে। Multi-class classification এর জন্য এটাই standard choice।
- **Optimizer (`adam`)**: loss কমানোর জন্য model এর weights (শুধু trainable অংশ) কীভাবে update করতে হবে সেটা ঠিক করে।
- **Metric (`accuracy`)**: training এর সময় কোন measurement দিয়ে বুঝবো model কতটা ভালো করছে।

---

## 🔄 Step 9: Data Augmentation — কম data থেকে বেশি শেখা বের করা

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
```

আমাদের dataset এ image সংখ্যা তুলনামূলক কম, তাই training data এর প্রতিটা image কে সামান্য পরিবর্তন করে (zoom, shear, horizontal flip) model কে দেখানো হচ্ছে, যাতে মনে হয় প্রতিবার একটা নতুন image দেখছে। এতে দুটো লাভ হয়:
- Model বেশি **robust** হয় (real world এ image সবসময় perfectly centered বা একই angle এ থাকে না)
- **Overfitting কমে**, কারণ model একই image বারবার হুবহু দেখে না

লক্ষণীয় ব্যাপার — এই augmentation শুধু **training data** তে করা হয়েছে (`train_datagen`)। Test data (`test_datagen`) তে শুধু `rescale` করা হয়েছে (0–1 এ normalize), কোনো augmentation না — কারণ testing সবসময় "real", unmodified data দিয়ে করতে হয়।

---

## 📥 Step 10: Folder থেকে সরাসরি Data Load করা

```python
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
```

`flow_from_directory` directly train আর test folder এর ভিতর ঢুকে, sub-folder এর নাম গুলোকে class label হিসেবে ধরে, image গুলো সঠিক size এ resize করে, আর ছোট ছোট batch (এখানে batch size 32) এ ভাগ করে model কে খাওয়ানোর জন্য প্রস্তুত করে দেয়। এর ফলে manually কোনো label array বানাতে হয়নি — পুরোটাই folder structure থেকে automatic।

---

## 🚀 Step 11: আসল Training — মাত্র নতুন Layer গুলো শেখা

```python
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
```

Model কে training data দিয়ে fit করানো হয়েছে, মোট **10 epochs** ধরে (মানে model পুরো training dataset টা 10 বার ঘুরে দেখেছে)। প্রতিটা epoch শেষে validation (test) data দিয়ে check করা হয়েছে model কতটা ভালো generalize করছে।

মনে রাখার বিষয় — এই training এ **শুধু নতুন যোগ করা Dense layer গুলোর weights update হচ্ছে**। VGG16 এর ভেতরের হাজার হাজার parameter অপরিবর্তিত (frozen) থেকেছে। এই কারণেই মাত্র 10 epoch আর অল্প data দিয়েও ভালো result পাওয়া সম্ভব হয়েছে।

---

## 📊 Step 12: ফলাফল Visualize করা

```python
import matplotlib.pyplot as plt
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
```

```python
# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
```

- যদি training loss কমতে থাকে কিন্তু validation loss বাড়তে থাকে → এটা **overfitting** এর signal
- যদি দুটোই ভালোভাবে কমতে থাকে আর কাছাকাছি থাকে → model ভালোভাবে শিখছে এবং generalize করছে

Test data দিয়ে final loss আর accuracy বের করতে:

```python
model.evaluate(test_set)
```

---

## 💾 Step 13: Model Save করা — যাতে বারবার Train করতে না হয়

```python
from tensorflow.keras.models import load_model

model.save('model_vgg16.h5')
```

পরে আবার load করতে:

```python
model = load_model("model_vgg16.h5")
```

একবার model ভালোভাবে train হয়ে গেলে, সেটা `.h5` ফরম্যাটে save করে রাখা হয়েছে। এর সুবিধা হলো — পরবর্তীতে নতুন কোনো image predict করতে হলে, পুরো training process আবার করতে হবে না, শুধু এই saved file টা load করলেই কাজ হবে।

---

## 🔮 Step 14: নতুন একটা Image দিয়ে Real Prediction করা

একটা সম্পূর্ণ নতুন X-ray image load করা হচ্ছে:

```python
from tensorflow.keras.preprocessing import image

img = "/content/Bones-data/test/Oblique fracture/43390tn_jpg.rf.e82c12a328a56cc66d5d828e638324be.jpg"
```

```python
img = image.load_img(img, target_size=(224,224))
img
```

Image টা numpy array তে convert করা হচ্ছে:

```python
x = image.img_to_array(img)
x
```

Image টা visually দেখতে:

```python
Z = plt.imread('/content/Bones-data/test/Oblique fracture/43390tn_jpg.rf.e82c12a328a56cc66d5d828e638324be.jpg')
plt.imshow(Z)
```

Array এর shape check করা হচ্ছে — `(224, 224, 3)` হওয়া উচিত:

```python
x.shape
```

Pixel value normalize করা হচ্ছে:

```python
x = x/255
```

Batch dimension যোগ আর VGG16-specific preprocessing:

```python
from keras.applications.vgg16 import preprocess_input
import numpy as np
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
```

প্রতিটা ধাপ গুরুত্বপূর্ণ, কারণ model যেভাবে training এর সময় data দেখেছিল, নতুন image কেও ঠিক একই format এ আনতে হবে:
1. **Resize** — 224×224, ঠিক training এর মতোই
2. **Array এ Convert** — model সংখ্যা বোঝে, image না
3. **Normalize** — 0–1 range এ আনা
4. **Batch Dimension** (`expand_dims`) — shape `(224, 224, 3)` থেকে `(1, 224, 224, 3)`, কারণ model সবসময় batch আশা করে
5. **VGG16 Preprocessing** (`preprocess_input`) — VGG16 training এর সময় যেভাবে data প্রসেস হয়েছিল, ঠিক সেই একই প্রসেসিং নতুন image এও apply করতে হয়, নাহলে model confuse হয়ে ভুল prediction দেবে

***`x = np.expand_dims(x, axis=0)` — কাজ কী?***

এই line টা array তে একটা নতুন dimension যোগ করে একদম শুরুতে (`axis=0`)।

**সমস্যা:** আমাদের image `x` এর shape ছিল `(224, 224, 3)` — মানে height, width, আর RGB channel। কিন্তু Keras model সবসময় input হিসেবে একটা **batch** আশা করে (একগুচ্ছ image), single image সরাসরি নেয় না। তাই expected shape হয় `(batch_size, 224, 224, 3)`।

**সমাধান:** `expand_dims` দিয়ে `axis=0` position এ একটা নতুন dimension (size 1) ঢুকিয়ে দেওয়া হয়:

- Before: `x.shape → (224, 224, 3)`
- After: `x.shape → (1, 224, 224, 3)`

এতে single image টা "1টা image সম্বলিত একটা batch" এ রূপান্তরিত হয়, যাতে `model.predict()` সেটা সঠিকভাবে accept করতে পারে।

**সহজ analogy:** ধরো তোমার কাছে একটা আপেল আছে, কিন্তু দোকানদার শুধু বাক্স নেয়, খুচরা আপেল নেয় না। তাই একটা আপেলকেই একটা বাক্সে ভরে দিলে — এখন বাক্সে একটাই আপেল, কিন্তু format টা এখন acceptable।

**না করলে কী হতো?** `model.predict(x)` কল করলে shape mismatch error আসতো, কারণ model 4-dimensional input আশা করে, কিন্তু পাচ্ছে 3-dimensional।

---

## 🏁 Step 15: চূড়ান্ত সিদ্ধান্ত — Prediction থেকে মানুষের বোঝার ভাষায়

```python
output = model.predict(img_data)
output
```

Model raw probability দেয় — যেমন `[0.0238, 0.9762]`। এর মানে:
- Class 0 (Oblique fracture) হওয়ার সম্ভাবনা মাত্র 2.38%
- Class 1 (Spiral Fracture) হওয়ার সম্ভাবনা 97.62%

```python
result = np.argmax(output, axis=1)
result
```

`argmax` সবচেয়ে বেশি probability যেই index এ আছে সেটা বের করে দেয় — এখানে `[1]`।

```python
if result[0] == 0:
    prediction = 'Oblique fracture'
    print(prediction)
else:
    prediction = 'Spiral Fracture'
    print(prediction)
```

Numeric label কে actual class name এ convert করে print করা হয়। এই notebook এর ক্ষেত্রে final output ছিল — **Spiral Fracture** ✅

---

## 🌟 পুরো গল্পের সারমর্ম — Transfer Learning এর Mental Model

> আমরা একজন **experienced expert** (VGG16, যে ImageNet এর কোটি কোটি image দেখে শিখেছে) কে ধার নিলাম। ওকে বললাম, "তোমার পুরনো জ্ঞান বদলাতে হবে না, ওটা যেমন আছে তেমনই থাকুক (**freeze**)। শুধু নতুন একটা ছোট্ট 'সিদ্ধান্ত নেওয়ার অংশ' (**Dense layers**) তোমার সাথে জুড়ে দিলাম, যেটা শুধু আমাদের নির্দিষ্ট সমস্যা — bone fracture চেনা — এই কাজেই ফোকাস করে শিখবে।" এরপর অল্প data আর অল্প সময় দিয়েই আমরা একটা কার্যকর classifier পেয়ে গেলাম, যেটা শুরু থেকে পুরো network train করলে অনেক বেশি data আর সময় লাগতো।

এই approach টাই আজকের দিনে medical imaging, object detection, face recognition — প্রায় সব জায়গায় ব্যাপকভাবে ব্যবহৃত হয়, কারণ এটা limited data নিয়েও powerful, production-ready model বানানোর সবচেয়ে বাস্তবসম্মত রাস্তা।

---

## 🧾 Tech Stack ব্যবহৃত এই Project এ
- **Base Model**: VGG16 (pretrained on ImageNet)
- **Framework**: TensorFlow / Keras
- **Technique**: Feature Extraction based Transfer Learning (frozen base + new trainable head)
- **Data Handling**: `ImageDataGenerator` with augmentation
- **Task**: Binary Image Classification (Oblique Fracture vs Spiral Fracture)
