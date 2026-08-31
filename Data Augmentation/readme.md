# Data Augmentation 

এই document এ Data Augmentation কে তিনটা দিক থেকে বোঝানো হয়েছে — **concept (theory)**, **সংখ্যার হিসাব (math)**, আর **আসল code**। উদ্দেশ্য হলো, তুমি যেন শুধু "এটা কাজ করে" না জেনে, বরং **ঠিক কীভাবে, কত পরিমাণে, আর কেন** কাজ করে — সেটা পুরোপুরি বুঝতে পারো।

---

## 🎯 Part 1: Theory — সমস্যা এবং সমাধান

### সমস্যা: অল্প Data, বড় Model

একটা Deep Learning model, বিশেষ করে CNN, ভালোভাবে শিখতে হলে সাধারণত হাজার হাজার/লক্ষ লক্ষ example দরকার হয়। কিন্তু বাস্তবে (যেমন medical X-ray dataset), আমাদের হাতে হয়তো মাত্র কয়েকশো image থাকে। এই situation এ সরাসরি train করলে model **Overfitting** এ ভোগে — training data মুখস্থ করে ফেলে, কিন্তু নতুন data তে fail করে।

### সমাধান: Data Augmentation

> Data Augmentation হলো এমন একটা কৌশল, যেখানে existing image গুলোকে সামান্য পরিবর্তন (zoom, shear, flip ইত্যাদি) করে, model কে প্রতিবার একটু ভিন্ন version দেখানো হয় — যাতে dataset এর physical size না বাড়িয়েও, model কার্যত অনেক বেশি variation থেকে শেখে।

এটা কাজ করে দুইটা স্তরে:
1. **Object তৈরি করা** (`ImageDataGenerator`) — কোন কোন transformation, কতটুকু মাত্রায় হবে, সেই নিয়ম লেখা
2. **Data supply করা** (`flow_from_directory`) — folder থেকে আসল image নিয়ে সেই নিয়ম অনুযায়ী প্রসেস করা, batch বানানো, model কে দেওয়া

---

## 🧮 Part 2: Math — সংখ্যার হিসাব

এখানেই সবচেয়ে বেশি ভুল বোঝাবুঝি হয়, তাই ধাপে ধাপে হিসাব করি।

### ধরে নিচ্ছি
- মোট training image সংখ্যা: **320টা**
- `batch_size = 32`

### ❌ ভুল ধারণা
"32টা image নেওয়া হবে, প্রতিটাতে shear/zoom/flip এর 4টা combination হবে, তাই মোট 32 × 4 = 128টা image একটা batch এ যাবে।"

**এটা সঠিক না।** Augmentation image সংখ্যা বাড়ায় না, প্রতিটা image কে শুধু বদলে দেয়।

### ✅ আসল হিসাব

**এক Batch এ কী হয়:**
```
Batch size = 32
→ প্রতি step এ ঠিক 32টা image model এ যায় (সংখ্যা অপরিবর্তিত)
→ প্রতিটা image আলাদাভাবে, randomly transform হয় (shear + zoom + flip এর একটা random মাত্রা)
→ Output: 32টা transformed image (32টাই, 128 না)
```

**এক Epoch এ কয়টা Step লাগে:**
```
Total images ÷ batch_size = steps per epoch
320 ÷ 32 = 10 steps
```

তাই কোডে যেটা লেখা হয়েছিল:
```python
steps_per_epoch = len(training_set)
```
এই `len(training_set)` স্বয়ংক্রিয়ভাবে হিসাব করে দেয় মোট কত step লাগবে (এখানে 10)।

**Randomness কীভাবে কাজ করে (প্রতিটা parameter এর range):**

| Parameter | Range | মানে |
|---|---|---|
| `shear_range=0.2` | 0% থেকে 20% এর মধ্যে random shear angle বাছাই হয় প্রতিটা image এর জন্য আলাদাভাবে | কোনো fixed মাত্রা না, প্রতিবার ভিন্ন |
| `zoom_range=0.2` | 80% থেকে 120% এর মধ্যে random zoom factor (±20%) | কখনো zoom in, কখনো zoom out |
| `horizontal_flip=True` | 50% probability তে flip হয়, 50% probability তে হয় না | Coin toss এর মতো — প্রতি image এ আলাদা সিদ্ধান্ত |

তাই batch এর প্রতিটা image এর জন্য transformation এর সিদ্ধান্ত হয় **independently, randomly** — কোনো fixed "4-combination" formula নেই।

### 📊 Multi-Epoch Effect — আসল লাভ কোথায়?

Physical dataset কখনো বাড়ছে না (320টাই থাকছে), কিন্তু **সময়ের সাথে (multiple epochs জুড়ে)** model একই image এর ভিন্ন ভিন্ন version দেখছে:

```
Epoch 1: Image_A → shear 12%, zoom 105%, flip না       (version 1)
Epoch 2: Image_A → shear 3%,  zoom 92%,  flip হ্যাঁ      (version 2)
Epoch 3: Image_A → shear 18%, zoom 110%, flip হ্যাঁ      (version 3)
...
Epoch 10: Image_A → shear 7%, zoom 88%,  flip না        (version 10)
```

যদি আমরা **10 epochs** train করি, তাহলে conceptually model প্রতিটা image এর **প্রায় 10টা ভিন্ন variation** দেখে ফেলেছে — যদিও dataset size বাড়েনি:

```
Effective "exposure" ≈ 320 images × 10 epochs = 3,200 বার image দেখা
(কিন্তু প্রতিটা বার সামান্য ভিন্ন রূপে — একদম হুবহু কপি না)
```

এটাই মূল গণিতগত সুবিধা — **storage বা actual data না বাড়িয়ে, effective training exposure বাড়ানো**।

---

## 💻 Part 3: Code — লাইন ধরে ধরে

### Step 1: Augmentation এর নিয়ম Define করা

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
```

**কী ঘটছে:** এই মুহূর্তে কোনো image touch হচ্ছে না। এটা শুধু rule/blueprint তৈরি করছে।

- `rescale=1./255` → pixel value কে 0–255 থেকে 0–1 এ normalize করার নিয়ম
- `shear_range=0.2` → maximum 20% পর্যন্ত shear (tilt) করার permission
- `zoom_range=0.2` → maximum ±20% zoom করার permission
- `horizontal_flip=True` → flip করার permission (on/off switch)
- `test_datagen` এ শুধু rescale — কারণ test data unbiased/unmodified রাখতে হয়

### Step 2: Folder থেকে Data Load এবং Augmentation Apply করা

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

**কী ঘটছে ধাপে ধাপে:**
1. `train_path` folder এ ঢোকে, sub-folder নাম গুলোকে class label হিসেবে নেয়
2. প্রতিটা image কে `(224, 224)` size এ resize করে
3. প্রতিটা image এর উপর, উপরে define করা নিয়ম অনুযায়ী **independent random transformation** apply করে
4. `batch_size=32` অনুযায়ী 32টা (transformed) image একসাথে group করে দেয়
5. `class_mode='categorical'` অনুযায়ী label কে one-hot vector এ convert করে

### Step 3: Training এ ব্যবহার

```python
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
```

- `epochs=10` → পুরো dataset 10 বার ঘোরানো হবে, প্রতিবার নতুন random augmentation সহ
- `steps_per_epoch=len(training_set)` → প্রতি epoch এ কতগুলো batch/step লাগবে সেটা automatically হিসাব হয় (`total_images / batch_size`)
- প্রতিটা epoch এ, প্রতিটা batch এ, প্রতিটা image **নতুনভাবে randomly augment** হয় — তাই model কখনো ঠিক একই version দুইবার দেখে না (উচ্চ সম্ভাবনায়)

---

## 🖼️ Part 4: এক নজরে সম্পূর্ণ Flow (Visual)

```
Dataset Folder (320 images)
        │
        ▼
train_datagen (rules: rescale, shear, zoom, flip)
        │
        ▼
flow_from_directory(batch_size=32)
        │
        ▼
   ┌─────────────────────────────┐
   │  Step 1: 32 images          │  ← প্রতিটা independently augmented
   │  Step 2: 32 images          │
   │  ...                        │
   │  Step 10: 32 images         │  ← 320/32 = 10 steps = 1 epoch সম্পূর্ণ
   └─────────────────────────────┘
        │
        ▼
   Epoch শেষে আবার প্রথম থেকে শুরু (Epoch 2)
   কিন্তু এবার প্রতিটা image আবার নতুন random augmentation পায়
        │
        ▼
   এভাবে 10 epochs → model প্রতিটা image এর ~10টা ভিন্ন version দেখে ফেলে
```

---

## ✅ মূল Takeaways

| ভুল ধারণা | আসল সত্য |
|---|---|
| Batch এ image সংখ্যা augmentation এর কারণে বাড়ে (32×4=128) | Batch size সবসময় fixed (32), augmentation শুধু ওই 32টাকে বদলে দেয়, সংখ্যা বাড়ায় না |
| প্রতিটা image এ shear+zoom+flip এর একটা "fixed 4-combination" set হয় | প্রতিটা transformation independently, randomly প্রয়োগ হয় — কোনো fixed combination formula নেই |
| Data Augmentation dataset এর physical size বাড়ায় | না, physical size (320) সবসময় same থাকে; শুধু **multi-epoch এর মধ্য দিয়ে effective exposure** বাড়ে |
| Test data তেও augmentation করা উচিত ভালো result এর জন্য | না, test data সবসময় unmodified রাখতে হয় fair evaluation এর জন্য |

**এক লাইনে সারমর্ম:** Data Augmentation কোনো "batch এর ভিতরে multiplication" না — এটা **epoch থেকে epoch এর মধ্যে randomness** যোগ করার একটা কৌশল, যাতে fixed সংখ্যক image থেকে model maximum variation শিখতে পারে।
