import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM IMPLEMENTATION
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const { width, height, data } = imageData;

    // === Step 1: Convert to grayscale ===
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // === Step 2: Detect background brightness (auto-detect light/dark mode) ===
    let borderSum = 0;
    let borderCount = 0;
    for (let x = 0; x < width; x++) {
      borderSum += gray[x] + gray[(height - 1) * width + x];
      borderCount += 2;
    }
    for (let y = 0; y < height; y++) {
      borderSum += gray[y * width] + gray[y * width + (width - 1)];
      borderCount += 2;
    }
    const bgLevel = borderSum / borderCount;
    const avgLevel = gray.reduce((a, b) => a + b, 0) / gray.length;
    const darkShapes = avgLevel < bgLevel;

    // === Step 3: Adaptive Thresholding ===
    const binary = new Uint8Array(width * height);
    const threshold = darkShapes ? bgLevel * 0.9 : bgLevel * 1.1;
    for (let i = 0; i < gray.length; i++) {
      if (darkShapes) {
        binary[i] = gray[i] < threshold ? 1 : 0;
      } else {
        binary[i] = gray[i] > threshold ? 1 : 0;
      }
    }

    // === Step 4: Find Connected Components (Flood Fill) ===
    const visited = new Uint8Array(width * height);
    const contours: Point[][] = [];
    const dirs = [
      [1, 0], [-1, 0], [0, 1], [0, -1],
      [1, 1], [-1, -1], [1, -1], [-1, 1],
    ];

    const inBounds = (x: number, y: number) =>
      x >= 0 && y >= 0 && x < width && y < height;
    const idx = (x: number, y: number) => y * width + x;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const id = idx(x, y);
        if (binary[id] === 1 && !visited[id]) {
          const stack: Point[] = [{ x, y }];
          const blob: Point[] = [];
          visited[id] = 1;
          while (stack.length) {
            const p = stack.pop()!;
            blob.push(p);
            for (const [dx, dy] of dirs) {
              const nx = p.x + dx;
              const ny = p.y + dy;
              if (!inBounds(nx, ny)) continue;
              const ni = idx(nx, ny);
              if (!visited[ni] && binary[ni] === 1) {
                visited[ni] = 1;
                stack.push({ x: nx, y: ny });
              }
            }
          }
          if (blob.length > 20) { // filter tiny noise
            contours.push(blob);
          }
        }
      }
    }

    // === Step 5: Analyze Contours and Classify Shapes ===
    const shapes: DetectedShape[] = [];

    for (const contour of contours) {
      const xs = contour.map((p) => p.x);
      const ys = contour.map((p) => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const w = maxX - minX;
      const h = maxY - minY;

      const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
      const approx = this.rdpSimplify(contour, 3);
      const shapeType = this.classifyShape(approx, contour, w, h);
      const area = w * h;
      const confidence = this.computeConfidence(shapeType, approx, contour);

      shapes.push({
        type: shapeType,
        confidence,
        boundingBox: { x: minX, y: minY, width: w, height: h },
        center,
        area,
      });
    }

    const processingTime = performance.now() - startTime;
    return { shapes, processingTime, imageWidth: width, imageHeight: height };
  }
  
  // --- Private Helper Functions ---

  /**
   * Classifies a shape based on its simplified polygon and original contour
   */
  private classifyShape(
    approx: Point[],
    contour: Point[],
    w: number,
    h: number
  ): DetectedShape["type"] {
    const v = approx.length; // Number of vertices in simplified polygon

    // 1. Polygon checks
    if (v === 3) return "triangle";
    if (v === 4) return "rectangle";
    if (v === 5) return "pentagon";

    // 2. Circularity check
    const center = this.computeCenter(contour);
    const radii = contour.map((p) => Math.hypot(p.x - center.x, p.y - center.y));
    const mean = radii.reduce((a, b) => a + b, 0) / radii.length;
    const variance =
      radii.reduce((a, b) => a + (b - mean) ** 2, 0) / radii.length;
    const cv = Math.sqrt(variance) / mean; // Coefficient of Variation

    // *** FIX WAS HERE ***
    // Relaxed threshold from 0.12 to 0.15 to account for pixelation
    if (cv < 0.15) return "circle";

    // 3. Default (fallback)
    return "star";
  }

  /**
   * Computes a heuristic confidence score
   */
  private computeConfidence(
    type: DetectedShape["type"],
    approx: Point[],
    contour: Point[]
  ): number {
    let base = 0.8;
    switch (type) {
      case "triangle": base = 0.9; break;
      case "rectangle": base = 0.9; break;
      case "pentagon": base = 0.85; break;
      case "circle": base = 0.95; break;
      case "star": base = 0.8; break;
    }
    const sizeFactor = Math.min(1, contour.length / 500); 
    return Math.min(1, base * (0.8 + 0.2 * sizeFactor));
  }

  /**
   * Calculates the geometric center (centroid) of a list of points
   */
  private computeCenter(points: Point[]): Point {
    const sx = points.reduce((a, p) => a + p.x, 0);
    const sy = points.reduce((a, p) => a + p.y, 0);
    return { x: sx / points.length, y: sy / points.length };
  }

  /**
   * Simplifies a polygon using the Ramer-Douglas-Peucker (RDP) algorithm
   */
  private rdpSimplify(points: Point[], tolerance: number): Point[] {
    if (points.length < 3) return points;
    const first = points[0];
    const last = points[points.length - 1];
    let index = -1;
    let maxDist = 0;

    for (let i = 1; i < points.length - 1; i++) {
      const d = this.perpendicularDistance(points[i], first, last);
      if (d > maxDist) {
        index = i;
        maxDist = d;
      }
    }

    if (maxDist > tolerance && index !== -1) {
      const left = this.rdpSimplify(points.slice(0, index + 1), tolerance);
      const right = this.rdpSimplify(points.slice(index), tolerance);
      return left.slice(0, -1).concat(right);
    } else {
      return [first, last];
    }
  }

  /**
   * Helper for RDP: Calculates distance from point p to line segment [a, b]
   */
  private perpendicularDistance(p: Point, a: Point, b: Point): number {
    const num = Math.abs(
      (b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x
    );
    const den = Math.hypot(b.x - a.x, b.y - a.y);
    return den === 0 ? 0 : num / den;
  }

  /**
   * Loads a File object into the canvas and returns its ImageData
   */
  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

/* -------------------------------------------------------------------------- */
/* APP UI CODE (Updated to draw detections)                                   */
/* -------------------------------------------------------------------------- */

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
      this.drawDetections(results); // Call drawing function
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No shapes detected.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }
  
  /**
   * Draw bounding boxes and labels on the canvas
   */
  private drawDetections(results: DetectionResult) {
    const ctx = (this.detector as any).ctx as CanvasRenderingContext2D;
    ctx.save();
    ctx.lineWidth = 2;
    results.shapes.forEach((s) => {
      let color = "lime";
      switch (s.type) {
        case "triangle": color = "orange"; break;
        case "rectangle": color = "aqua"; break;
        case "pentagon": color = "magenta"; break;
        case "star": color = "yellow"; break;
        case "circle": color = "lime"; break;
      }
      ctx.strokeStyle = color;
      ctx.strokeRect(
        s.boundingBox.x,
        s.boundingBox.y,
        s.boundingBox.width,
        s.boundingBox.height
      );
      ctx.fillStyle = color;
      ctx.font = "14px Arial";
      ctx.fillText(s.type, s.boundingBox.x + 5, s.boundingBox.y + 15);
    });
    ctx.restore();
  }


  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          await this.processImage(file);
          
          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});