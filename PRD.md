# Product Requirements Document
## UI Motion Reverse Engineer

**Version:** 0.1  
**Status:** MVP Definition

---

# 1. Product Overview

UI Motion Reverse Engineer adalah web application yang mengubah screen recording dari sebuah UI interaction menjadi **structured motion specification dalam bentuk teks**.

User merekam interaksi dengan sebuah website atau aplikasi, misalnya:

- hover button
- hover card
- membuka dropdown
- membuka modal
- sidebar expand/collapse
- accordion
- page transition
- navigation animation
- drag interaction
- scroll animation
- loading state
- micro-interaction

User kemudian meng-upload video tersebut.

System menganalisis perubahan visual sepanjang timeline dan menghasilkan deskripsi teknis yang cukup detail sehingga dapat diberikan kepada coding LLM seperti ChatGPT, Claude, Gemini, atau coding agent untuk mereplikasi interaction dan motion tersebut.

Core transformation:

```text
Screen Recording
       ↓
Video Analysis
       ↓
UI State Detection
       ↓
Interaction Detection
       ↓
Motion Analysis
       ↓
Structured Motion Specification
       ↓
LLM-ready Prompt
```

---

# 2. Problem

Developer sering menemukan UI atau micro-interaction yang ingin direplikasi.

Namun motion sulit dijelaskan hanya melalui screenshot.

Screenshot hanya menunjukkan:

```text
STATE A
```

dan

```text
STATE B
```

tetapi tidak menjelaskan bagaimana transisi terjadi.

Contohnya sebuah card:

```text
Default
   ↓
cursor enters
   ↓
card moves upward
   ↓
shadow increases
   ↓
image scales
   ↓
title changes color
```

Informasi penting lainnya juga hilang:

```text
duration
delay
easing
direction
distance
scale
opacity
transform origin
stagger
trigger
sequence
```

Akibatnya user harus menjelaskan motion secara manual kepada LLM.

Contoh prompt yang kurang presisi:

> "Bikin card ini naik sedikit dan gambarnya zoom ketika hover."

Hasil implementasi kemudian sangat bergantung pada interpretasi LLM.

---

# 3. Product Goal

Mengubah:

```text
"I want this interaction."
```

menjadi:

```text
"When the cursor enters the card boundary,
the card translates vertically from 0px to -8px
over approximately 280ms using an ease-out curve.

At the same time, the image scales from 1.00 to approximately 1.06
with transform-origin centered.

The card shadow gradually increases from
0 4px 12px rgba(0,0,0,0.08)
to approximately
0 12px 32px rgba(0,0,0,0.16).

When the cursor leaves,
all properties transition back to their initial values
over approximately 220ms."
```

Output harus cukup teknis sehingga coding LLM dapat menggunakan deskripsi tersebut sebagai implementation specification.

---

# 4. Target User

Primary users:

**Frontend Developers**

Developer menemukan interaction menarik dari website lain dan ingin mempelajari atau mereplikasinya.

**UI Engineers**

Engineer membutuhkan referensi motion yang lebih presisi daripada video.

**UI/UX Designers**

Designer ingin mengubah motion reference menjadi specification untuk developer.

**AI-assisted Developers**

User menggunakan coding LLM/agent dan membutuhkan prompt yang menjelaskan reference UI secara detail.

---

# 5. Primary User Flow

## Step 1 — Capture

User membuka UI yang ingin dianalisis.

User melakukan screen recording.

Recommended recording:

```text
2–15 seconds
```

User melakukan satu interaction utama.

Contoh:

```text
cursor outside card

↓

cursor enters card

↓

hover animation plays

↓

hold 1 second

↓

cursor leaves card

↓

return animation plays
```

Untuk MVP, user dianjurkan melakukan **satu interaction per recording**.

---

## Step 2 — Upload

User membuka web application.

UI:

```text
┌───────────────────────────────────────────┐
│                                           │
│       Drop your UI recording here         │
│                                           │
│       MP4 / MOV / WebM                     │
│                                           │
│       Recommended: 2–15 seconds           │
│                                           │
│             [ Upload Video ]               │
│                                           │
└───────────────────────────────────────────┘
```

---

# 6. Analysis Flow

Setelah upload, video diproses melalui beberapa tahap.

```text
VIDEO
 │
 ├── Metadata extraction
 │
 ├── Frame extraction
 │
 ├── Scene / UI state analysis
 │
 ├── Cursor tracking
 │
 ├── Interaction detection
 │
 ├── Element tracking
 │
 ├── Motion estimation
 │
 ├── Visual property estimation
 │
 └── Motion specification generation
```

---

# 7. Stage 1 — Video Preprocessing

System membaca:

```text
resolution
frame rate
duration
frame count
```

Contoh:

```json
{
  "resolution": "1920x1080",
  "fps": 60,
  "duration": 6.4
}
```

Untuk motion analysis, jangan hanya mengambil satu frame per detik.

Sampling harus lebih tinggi karena micro-animation biasanya berlangsung hanya:

```text
100–500ms
```

MVP recommendation:

```text
source video: ideally 60 FPS

analysis sampling:
15–30 FPS
```

Untuk interaction yang terdeteksi, system dapat melakukan second-pass dengan frame density lebih tinggi.

---

# 8. Stage 2 — Cursor Tracking

Cursor sangat penting karena membantu menentukan **trigger interaction**.

System mencoba mendeteksi:

```text
cursor position
cursor movement
cursor enters element
cursor leaves element
cursor click
cursor drag
```

Contoh timeline:

```text
0.00s cursor outside card

1.12s cursor enters card

1.18s visual change starts

1.46s animation settles

2.80s cursor leaves

2.84s reverse animation starts

3.08s animation settles
```

Dari sini system dapat menyimpulkan:

```text
interaction_type: hover
```

---

# 9. Stage 3 — UI State Detection

System mencari state stabil sebelum dan sesudah interaction.

Contoh:

```text
STATE A

Card:
x: 320
y: 440
width: 360
height: 420

Image:
scale: 1.00

Title:
color: #111111
```

Kemudian:

```text
STATE B

Card:
x: 320
y: 432
width: 360
height: 420

Image:
scale: 1.06

Title:
color: #5252FF
```

Perbedaan:

```text
Card translateY = -8px

Image scale =
1.00 → 1.06

Title color =
#111111 → #5252FF
```

---

# 10. Stage 4 — Element Detection

System mengidentifikasi hierarchy visual.

Contoh:

```text
Card
├── Image Container
│   └── Image
│
├── Content
│   ├── Category
│   ├── Title
│   └── Description
│
└── Arrow Icon
```

LLM/Vision model membantu memberikan semantic label terhadap element.

---

# 11. Stage 5 — Motion Tracking

Untuk setiap element, system membandingkan perubahan antar-frame.

Properties yang dianalisis:

### Geometry

```text
x
y
width
height
```

### Transform

```text
translateX
translateY
scaleX
scaleY
rotation
```

### Visual

```text
opacity
color
background
border
border-radius
shadow
blur
```

### Content

```text
text changes
icon changes
image changes
```

---

# 12. Stage 6 — Timing Analysis

System mengestimasi:

```text
start time
end time
duration
delay
```

Contoh:

```text
Card translate

start: 1.18s
end: 1.46s

duration ≈ 280ms
```

---

# 13. Stage 7 — Easing Estimation

Dengan tracking position sepanjang timeline:

```text
time      translateY

0ms       0px
50ms     -2.8px
100ms    -5.1px
150ms    -6.7px
200ms    -7.6px
280ms    -8px
```

System mengestimasi curve yang paling mendekati.

Output bisa berupa:

```text
ease-out
```

atau jika confidence cukup tinggi:

```css
cubic-bezier(0.16, 1, 0.3, 1)
```

System harus menandai ini sebagai **estimated value**, bukan exact CSS asli.

---

# 14. Stage 8 — Motion Relationship

System mendeteksi apakah animation terjadi:

```text
simultaneously
sequentially
staggered
```

Contoh:

```text
0ms
Card translate starts

0ms
Shadow transition starts

30ms
Image scale starts

80ms
Arrow moves

280ms
Card settles

320ms
Image settles
```

Output:

```text
Card and shadow animate simultaneously.

Image scaling begins approximately 30ms later.

Arrow movement begins approximately 80ms after
the primary hover transition.
```

---

# 15. Intermediate Representation

Jangan langsung mengubah video menjadi prose.

System harus memiliki structured representation terlebih dahulu.

Contoh:

```json
{
  "interaction": {
    "type": "hover",
    "target": "product-card"
  },

  "initial_state": {
    "card": {
      "translateY": 0
    },

    "image": {
      "scale": 1
    },

    "arrow": {
      "translateX": 0,
      "opacity": 0.7
    }
  },

  "active_state": {
    "card": {
      "translateY": -8
    },

    "image": {
      "scale": 1.06
    },

    "arrow": {
      "translateX": 4,
      "opacity": 1
    }
  },

  "transitions": [
    {
      "element": "card",
      "property": "transform.translateY",
      "from": "0px",
      "to": "-8px",
      "duration_ms": 280,
      "easing": "ease-out",
      "confidence": 0.89
    }
  ]
}
```

Structured representation ini menjadi **source of truth**.

---

# 16. Confidence System

System tidak boleh berpura-pura mengetahui property asli dari website.

Video hanya memungkinkan system melakukan estimation.

Setiap measurement memiliki confidence:

```text
High confidence
Medium confidence
Low confidence
```

Contoh:

```text
translateY:
-8px
confidence: 94%

duration:
280ms
confidence: 91%

scale:
1.06
confidence: 87%

easing:
cubic-bezier(0.16, 1, 0.3, 1)
confidence: 61%

shadow:
estimated
confidence: 48%
```

---

# 17. Analysis Result UI

Setelah processing:

```text
┌──────────────────────────────────────────────────┐
│ Original Video                                   │
│                                                  │
│             [ Video Player ]                     │
│                                                  │
├──────────────────────────────────────────────────┤
│ Detected Interaction                             │
│                                                  │
│ Trigger        Hover                             │
│ Target         Product Card                      │
│ Duration       ~280ms                            │
│ Direction      Forward + Reverse                 │
├──────────────────────────────────────────────────┤
│ Motion Timeline                                  │
│                                                  │
│ Card       █████████████                         │
│ Shadow     █████████████                         │
│ Image        ██████████████                      │
│ Arrow          █████████                         │
├──────────────────────────────────────────────────┤
│ Motion Specification                            │
│                                                  │
│ [Technical] [LLM Prompt] [JSON]                  │
└──────────────────────────────────────────────────┘
```

---

# 18. Output Mode #1 — Technical Description

Example:

```text
INTERACTION

Type:
Hover interaction

Target:
Entire product card

Trigger:
Animation begins when pointer enters card boundary.

CARD

Initial:
translateY(0px)

Hover:
translateY(-8px)

Duration:
approximately 280ms

Easing:
ease-out

IMAGE

Initial:
scale(1)

Hover:
scale(1.06)

Transform origin:
center center

Duration:
approximately 320ms

Delay:
approximately 30ms

SHADOW

Shadow becomes progressively larger and darker
during the hover transition.

ARROW

Arrow translates approximately 4px horizontally
while opacity increases from approximately
0.7 to 1.
```

---

# 19. Output Mode #2 — LLM Prompt

Ini adalah output utama MVP.

System menghasilkan prompt yang bisa langsung di-copy.

Example:

```text
Recreate the attached/reference UI interaction as follows.

STRUCTURE

Create a product card containing:
- image
- category label
- title
- description
- arrow icon

INTERACTION

The entire card acts as the hover target.

When the pointer enters the card:

1. Move the entire card upward approximately 8px.

2. Animate this movement over approximately 280ms
using an ease-out curve.

3. Simultaneously increase the card shadow.

4. Scale the image from 1.00 to approximately 1.06.

5. Start the image animation approximately 30ms
after the card animation begins.

6. Move the arrow approximately 4px to the right
and increase its opacity.

The animation should feel smooth and responsive,
not springy.

When the pointer leaves the card, reverse all
properties smoothly to their original states.

Do not invent additional animations that are not
described above.
```

Button:

```text
[ Copy Prompt ]
```

---

# 20. Output Mode #3 — JSON

Advanced users dapat mengambil raw motion specification.

```text
[ Copy JSON ]
```

Use cases:

```text
API
automation
agent workflow
dataset creation
fine-tuning
evaluation
```

---

# 21. Optional Output — CSS Suggestion

System dapat menghasilkan approximate implementation:

```css
.card {
  transition:
    transform 280ms ease-out,
    box-shadow 280ms ease-out;
}

.card:hover {
  transform: translateY(-8px);
}

.card img {
  transition:
    transform 320ms ease-out 30ms;
}

.card:hover img {
  transform: scale(1.06);
}
```

Important:

CSS merupakan **suggested implementation**, bukan klaim bahwa original UI menggunakan CSS tersebut.

---

# 22. Recommended Technical Architecture

```text
                    VIDEO UPLOAD
                         │
                         ▼
                       FFmpeg
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        FRAME PIPELINE        VIDEO METADATA
              │
              ▼
       Cursor Detection
              │
              ▼
       UI Segmentation
              │
              ▼
       Element Tracking
              │
              ▼
        Motion Tracking
              │
              ▼
      Property Estimation
              │
              ▼
       Timeline Builder
              │
              ▼
         Vision / LLM
              │
              ▼
      Semantic Interpretation
              │
              ▼
       MOTION SPEC JSON
              │
        ┌─────┼─────────┐
        ▼     ▼         ▼
      Text   Prompt     JSON
```

---

# 23. Important Architecture Decision

Vision LLM sebaiknya **tidak menjadi satu-satunya analyzer**.

Jangan hanya:

```text
Video
 ↓
LLM
 ↓
"Describe this animation"
```

karena hasilnya cenderung subjektif dan sulit mendapatkan angka konsisten.

Gunakan dua layer.

### Layer A — Measurement

Computer vision / deterministic processing menghitung:

```text
pixel movement
bounding box
timing
scale
cursor position
frame differences
```

### Layer B — Interpretation

Vision LLM menentukan:

```text
"This appears to be a card."

"This change was probably triggered by hover."

"The image is inside the card."

"The interaction resembles a hover micro-animation."
```

Kemudian:

```text
MEASUREMENT
+
SEMANTIC INTERPRETATION
        ↓
MOTION SPEC
```

---

# 24. Suggested Technology

Frontend:

```text
Next.js
React
Tailwind
```

Backend:

```text
Python
FastAPI
```

Video processing:

```text
FFmpeg
OpenCV
```

Computer vision:

```text
OpenCV
Optical Flow
Feature Tracking
```

Vision understanding:

```text
Multimodal LLM
```

Storage:

```text
S3-compatible object storage
```

Database:

```text
PostgreSQL
```

Queue untuk video processing:

```text
Redis
+
worker
```

---

# 25. MVP Scope

MVP **tidak perlu memahami semua jenis motion**.

Fokus pada:

```text
Hover
Click
Simple expand/collapse
Dropdown
Modal
Button micro-interaction
Card interaction
```

Properties:

```text
translate
scale
opacity
color
shadow
border-radius
```

Output:

```text
Technical description
LLM-ready prompt
JSON
```

Input:

```text
MP4
MOV
WebM

max 15 seconds
```

---

# 26. Out of Scope — MVP

Jangan dulu mencoba:

```text
complex 3D animation
WebGL
canvas animation
physics reconstruction
long page recordings
multi-page workflows
complex drag-and-drop
precise spring parameter reconstruction
exact source-code reconstruction
```

Masalah tersebut bisa ditambahkan setelah motion analysis dasar reliable.

---

# 27. Recording Guidelines

Untuk meningkatkan accuracy, application memberikan instruction sebelum upload:

```text
For best results:

1. Keep the recording short.

2. Record at 60 FPS when possible.

3. Keep the cursor visible.

4. Start recording before performing the interaction.

5. Wait approximately one second before interacting.

6. Perform one interaction.

7. Wait until the animation finishes.

8. Return the UI to its initial state when possible.
```

Contoh hover:

```text
Initial state
   ↓
wait
   ↓
cursor enters
   ↓
animation
   ↓
wait
   ↓
cursor exits
   ↓
reverse animation
   ↓
wait
```

Ini membuat state dan timing jauh lebih mudah dianalisis.

---

# 28. Key Product Principle

Product tidak mencoba menjawab:

> "Apa source code website ini?"

Product mencoba menjawab:

> "Perubahan visual apa yang terjadi, kapan perubahan tersebut terjadi, bagaimana element bergerak, dan bagaimana interaction tersebut dapat direplikasi?"

Perbedaan ini penting karena screen recording tidak memberikan akses terhadap:

```text
DOM
CSS
JavaScript
animation library
original easing
original design tokens
```

Semua nilai yang berasal dari visual analysis harus dianggap sebagai **visual reconstruction / estimation**.

---

# 29. Success Metrics

Primary metric:

**Replication usefulness**

Apakah output bisa diberikan kepada coding LLM dan menghasilkan motion yang visually close dengan reference?

Secondary metrics:

```text
Interaction detection accuracy

Motion property accuracy

Timing error

Position / scale error

Prompt usefulness

Copy → implementation success rate
```

Long-term metric:

```text
Video
 ↓
Generated Specification
 ↓
Coding Agent
 ↓
Generated UI
 ↓
Visual Comparison
 ↓
Similarity Score
```

Loop ini nantinya bisa menjadi basis automated evaluation.

---

# 30. Future Vision

V2 dapat menerima:

```text
Video + Screenshot
```

dan menghasilkan:

```text
UI structure
+
visual styling
+
motion specification
```

V3:

```text
Video
 ↓
UI Reconstruction
 ↓
React Component
 ↓
Visual Comparison
 ↓
Automatic Correction
 ↓
Near-identical UI
```

Ultimate workflow:

```text
Record interaction
       ↓
Upload
       ↓
Analyze
       ↓
Generate
       ↓
Preview
       ↓
Compare against reference
       ↓
Auto-correct
       ↓
Export React / CSS
```

Pada titik tersebut produknya berkembang dari **video-to-motion-description** menjadi **visual UI reverse-engineering agent**.
