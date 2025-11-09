# 🔷 Shape Detection - Flam Project

An intelligent web-based **Shape Detection System** built with TypeScript, HTML5 Canvas, and Vite.  
This project detects and classifies geometric shapes such as **circles, triangles, rectangles, pentagons, and stars** from uploaded or preloaded test images — directly in the browser, using pure image processing algorithms (no external CV libraries).

---

## 🚀 Live Demo
👉 [https://SriBhuvanaSankar.github.io/Shape-detection-Flam](https://SriBhuvanaSankar.github.io/Shape-detection-Flam) *(if deployed with GitHub Pages)*

---

## 🧠 Overview
The project analyzes an image’s pixel data, performs:
- Grayscale conversion  
- Adaptive thresholding  
- Flood-fill contour detection  
- Polygon approximation  
- Shape classification (vertex analysis & radius variance)  

and returns:
- Shape type  
- Confidence score  
- Bounding box  
- Area and center coordinates  

All results are visualized on a canvas overlay.

---

## 🧩 Features
✅ Detects **five major shapes:** circle, triangle, rectangle, pentagon, star  
✅ Works on **colored or grayscale** images  
✅ Supports **light or dark backgrounds** (auto polarity detection)  
✅ Draws bounding boxes and shape labels directly on canvas  
✅ Instant feedback for test and uploaded images  
✅ No external ML or OpenCV — uses **pure browser-native math**  

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-------------|
| Frontend | TypeScript + HTML5 + CSS |
| Build Tool | [Vite](https://vitejs.dev/) |
| Image Processing | Canvas API |
| Algorithms | Custom geometric + contour-based shape detection |
| UI Components | Native DOM & CSS |

---

## ⚙️ Setup Instructions

### Prerequisites
- Node.js (v16 or higher)
- npm (v8 or higher)

### Installation
```bash
# Clone your repository
git clone https://github.com/SriBhuvanaSankar/Shape-detection-Flam.git
cd Shape-detection-Flam

# Install dependencies
npm install

# Start the development server
npm run dev
