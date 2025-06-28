#!/usr/bin/env python3
"""
Robustness Evaluation Script for Grid-based Watermarking

This script evaluates the resistance of the grid-based watermarking strategy
against various types of attacks including:
- Geometric attacks (rotation, scaling, cropping)
- Signal processing attacks (noise, blur, compression)
- Filtering attacks (median, gaussian, bilateral)
- Color space attacks (brightness, contrast adjustment)

Author: Liam Huang
Date: 2024
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Callable
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent))
import fdwm


class AttackEvaluator:
    """Evaluator for watermark robustness against various attacks."""

    def __init__(self, host_path: str, watermark_path: str, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.

        Parameters
        ----------
        host_path : str
            Path to the host image
        watermark_path : str
            Path to the watermark image
        output_dir : str
            Directory to save evaluation results
        """
        self.host_path = Path(host_path)
        self.watermark_path = Path(watermark_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load images
        self.host_img = cv2.imread(str(self.host_path), cv2.IMREAD_GRAYSCALE)
        self.watermark_img = cv2.imread(str(self.watermark_path), cv2.IMREAD_GRAYSCALE)

        # Embedding parameters
        self.strength = 50000.0
        self.scale = 0.25
        self.grid_m = 3
        self.grid_n = 3

        # Results storage
        self.results = {}

    def embed_watermark(self) -> str:
        """Embed watermark and return path to watermarked image."""
        watermarked_path = self.output_dir / "watermarked.png"

        out_path, metrics = fdwm.embed(
            host_path=str(self.host_path),
            watermark_path=str(self.watermark_path),
            output_path=str(watermarked_path),
            strength=self.strength,
            scale=self.scale,
            grid_m=self.grid_m,
            grid_n=self.grid_n,
        )

        print(f"✅ Watermark embedded successfully")
        print(f"   Mean pixel diff: {metrics['mean_pixel_diff']:.2f}")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")

        return str(watermarked_path)

    def extract_watermark(self, attacked_path: str) -> np.ndarray:
        """Extract watermark from attacked image."""
        extracted = fdwm.extract(
            watermarked_path=attacked_path,
            strength=self.strength,
            scale=self.scale,
            grid_m=self.grid_m,
            grid_n=self.grid_n,
        )
        return extracted

    def calculate_similarity(self, original: np.ndarray, extracted: np.ndarray) -> Dict[str, float]:
        """Calculate similarity metrics between original and extracted watermarks."""
        # Resize original to match extracted size
        original_resized = cv2.resize(original, extracted.shape[::-1], interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        original_norm = original_resized.astype(np.float32) / 255.0
        extracted_norm = extracted.astype(np.float32) / 255.0

        # Calculate metrics
        # 1. Pearson correlation coefficient
        corr = np.corrcoef(original_norm.flatten(), extracted_norm.flatten())[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # 2. Mean Squared Error (MSE)
        mse = np.mean((original_norm - extracted_norm) ** 2)

        # 3. Peak Signal-to-Noise Ratio (PSNR)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        # 4. Structural Similarity Index (SSIM-like)
        mu1 = np.mean(original_norm)
        mu2 = np.mean(extracted_norm)
        sigma1 = np.std(original_norm)
        sigma2 = np.std(extracted_norm)
        sigma12 = np.mean((original_norm - mu1) * (extracted_norm - mu2))

        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

        return {
            'correlation': corr,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        }

    def apply_attack(self, img: np.ndarray, attack_name: str, **kwargs) -> np.ndarray:
        """Apply a specific attack to the image."""
        attacked = img.copy()

        if attack_name == "gaussian_noise":
            # Add Gaussian noise
            noise_level = kwargs.get('noise_level', 0.1)
            noise = np.random.normal(0, noise_level * 255, img.shape)
            attacked = np.clip(attacked.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif attack_name == "salt_pepper_noise":
            # Add salt and pepper noise
            noise_ratio = kwargs.get('noise_ratio', 0.05)
            num_noise_pixels = int(noise_ratio * img.size)
            for _ in range(num_noise_pixels):
                i = random.randint(0, img.shape[0] - 1)
                j = random.randint(0, img.shape[1] - 1)
                attacked[i, j] = random.choice([0, 255])

        elif attack_name == "gaussian_blur":
            # Apply Gaussian blur
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 1.0)
            attacked = cv2.GaussianBlur(attacked, (kernel_size, kernel_size), sigma)

        elif attack_name == "median_filter":
            # Apply median filter
            kernel_size = kwargs.get('kernel_size', 5)
            attacked = cv2.medianBlur(attacked, kernel_size)

        elif attack_name == "rotation":
            # Rotate image
            angle = kwargs.get('angle', 5)
            height, width = img.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            attacked = cv2.warpAffine(attacked, rotation_matrix, (width, height))

        elif attack_name == "scaling":
            # Scale image
            scale_factor = kwargs.get('scale_factor', 0.8)
            height, width = img.shape
            new_height, new_width = int(height * scale_factor), int(width * scale_factor)
            attacked = cv2.resize(attacked, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            # Resize back to original size
            attacked = cv2.resize(attacked, (width, height), interpolation=cv2.INTER_LINEAR)

        elif attack_name == "cropping":
            # Crop image
            crop_ratio = kwargs.get('crop_ratio', 0.1)
            height, width = img.shape
            crop_pixels = int(min(height, width) * crop_ratio)
            attacked = attacked[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
            # Resize back to original size
            attacked = cv2.resize(attacked, (width, height), interpolation=cv2.INTER_LINEAR)

        elif attack_name == "brightness_adjustment":
            # Adjust brightness
            factor = kwargs.get('factor', 1.2)
            attacked = np.clip(attacked.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        elif attack_name == "contrast_adjustment":
            # Adjust contrast
            factor = kwargs.get('factor', 1.5)
            mean_val = np.mean(attacked)
            attacked = np.clip((attacked.astype(np.float32) - mean_val) * factor + mean_val, 0, 255).astype(np.uint8)

        elif attack_name == "jpeg_compression":
            # Simulate JPEG compression
            quality = kwargs.get('quality', 50)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_img = cv2.imencode('.jpg', attacked, encode_param)
            attacked = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)

        elif attack_name == "sharpening":
            # Apply sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            attacked = cv2.filter2D(attacked, -1, kernel)
            attacked = np.clip(attacked, 0, 255)

        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        return attacked

    def evaluate_attack(self, attack_name: str, attack_params: Dict, watermarked_path: str) -> Dict:
        """Evaluate resistance against a specific attack."""
        print(f"\n🔍 Evaluating {attack_name} attack...")

        # Load watermarked image
        watermarked_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)

        # Apply attack
        attacked_img = self.apply_attack(watermarked_img, attack_name, **attack_params)

        # Save attacked image
        attacked_path = self.output_dir / f"attacked_{attack_name}.png"
        cv2.imwrite(str(attacked_path), attacked_img)

        # Extract watermark from attacked image
        extracted = self.extract_watermark(str(attacked_path))

        # Calculate similarity
        similarity = self.calculate_similarity(self.watermark_img, extracted)

        # Save extracted watermark
        extracted_path = self.output_dir / f"extracted_{attack_name}.png"
        cv2.imwrite(str(extracted_path), extracted)

        print(f"   Correlation: {similarity['correlation']:.3f}")
        print(f"   SSIM: {similarity['ssim']:.3f}")
        print(f"   PSNR: {similarity['psnr']:.2f} dB")

        return similarity

    def run_evaluation(self) -> Dict:
        """Run comprehensive robustness evaluation."""
        print("🚀 Starting Grid-based Watermarking Robustness Evaluation")
        print("=" * 60)

        # Step 1: Embed watermark
        print("\n📝 Step 1: Embedding watermark...")
        watermarked_path = self.embed_watermark()

        # Step 2: Define attacks to test
        attacks = {
            # Noise attacks
            "gaussian_noise": {"noise_level": 0.1},
            "salt_pepper_noise": {"noise_ratio": 0.05},

            # Filtering attacks
            "gaussian_blur": {"kernel_size": 5, "sigma": 1.0},
            "median_filter": {"kernel_size": 5},
            "sharpening": {},

            # Geometric attacks
            "rotation": {"angle": 5},
            "scaling": {"scale_factor": 0.8},
            "cropping": {"crop_ratio": 0.1},

            # Signal processing attacks
            "brightness_adjustment": {"factor": 1.2},
            "contrast_adjustment": {"factor": 1.5},
            "jpeg_compression": {"quality": 50},
        }

        # Step 3: Evaluate each attack
        print(f"\n🛡️ Step 2: Evaluating {len(attacks)} attacks...")
        for attack_name, attack_params in attacks.items():
            self.results[attack_name] = self.evaluate_attack(attack_name, attack_params, watermarked_path)

        # Step 4: Generate summary report
        self.generate_report()

        return self.results

    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        print("\n📊 Step 3: Generating evaluation report...")

        # Calculate average metrics
        avg_correlation = np.mean([r['correlation'] for r in self.results.values()])
        avg_ssim = np.mean([r['ssim'] for r in self.results.values()])
        avg_psnr = np.mean([r['psnr'] for r in self.results.values()])

        # Find best and worst attacks
        correlations = [(name, r['correlation']) for name, r in self.results.items()]
        correlations.sort(key=lambda x: x[1], reverse=True)

        best_attack = correlations[0]
        worst_attack = correlations[-1]

        # Generate report
        report_path = self.output_dir / "robustness_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Grid-based Watermarking Robustness Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Evaluation Parameters:\n")
            f.write(f"  - Strength: {self.strength}\n")
            f.write(f"  - Scale: {self.scale}\n")
            f.write(f"  - Grid size: {self.grid_m}×{self.grid_n}\n")
            f.write(f"  - Host image: {self.host_path.name}\n")
            f.write(f"  - Watermark image: {self.watermark_path.name}\n\n")

            f.write("Attack Results:\n")
            f.write("-" * 40 + "\n")
            for attack_name, metrics in self.results.items():
                f.write(f"{attack_name:20s}: Corr={metrics['correlation']:.3f}, "
                       f"SSIM={metrics['ssim']:.3f}, PSNR={metrics['psnr']:.2f}dB\n")

            f.write(f"\nSummary Statistics:\n")
            f.write(f"  - Average Correlation: {avg_correlation:.3f}\n")
            f.write(f"  - Average SSIM: {avg_ssim:.3f}\n")
            f.write(f"  - Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"  - Best Attack: {best_attack[0]} (Corr={best_attack[1]:.3f})\n")
            f.write(f"  - Worst Attack: {worst_attack[0]} (Corr={worst_attack[1]:.3f})\n")

            f.write(f"\nRobustness Assessment:\n")
            if avg_correlation > 0.8:
                f.write("  - Overall: EXCELLENT robustness\n")
            elif avg_correlation > 0.6:
                f.write("  - Overall: GOOD robustness\n")
            elif avg_correlation > 0.4:
                f.write("  - Overall: MODERATE robustness\n")
            else:
                f.write("  - Overall: POOR robustness\n")

        # Print summary to console
        print(f"\n📈 Evaluation Summary:")
        print(f"   Average Correlation: {avg_correlation:.3f}")
        print(f"   Average SSIM: {avg_ssim:.3f}")
        print(f"   Average PSNR: {avg_psnr:.2f} dB")
        print(f"   Best Attack: {best_attack[0]} (Corr={best_attack[1]:.3f})")
        print(f"   Worst Attack: {worst_attack[0]} (Corr={worst_attack[1]:.3f})")
        print(f"\n📄 Detailed report saved to: {report_path}")

        # Generate visualization
        self.generate_visualization()

    def generate_visualization(self):
        """Generate visualization of attack results."""
        try:
            import matplotlib.pyplot as plt

            # Prepare data
            attack_names = list(self.results.keys())
            correlations = [self.results[name]['correlation'] for name in attack_names]
            ssim_scores = [self.results[name]['ssim'] for name in attack_names]

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Correlation plot
            bars1 = ax1.bar(attack_names, correlations, color='skyblue', alpha=0.7)
            ax1.set_title('Correlation Coefficient by Attack Type', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Correlation Coefficient')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars1, correlations):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)

            # SSIM plot
            bars2 = ax2.bar(attack_names, ssim_scores, color='lightcoral', alpha=0.7)
            ax2.set_title('SSIM Score by Attack Type', fontsize=14, fontweight='bold')
            ax2.set_ylabel('SSIM Score')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars2, ssim_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()

            # Save plot
            plot_path = self.output_dir / "robustness_visualization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {plot_path}")

        except ImportError:
            print("⚠️  matplotlib not available, skipping visualization")


def main():
    """Main function to run the evaluation."""
    # Check if required files exist
    host_path = "tmp_extraction/host.png"
    watermark_path = "tmp_extraction/wm.png"

    if not Path(host_path).exists():
        print(f"❌ Host image not found: {host_path}")
        print("Please run tests first to generate test images.")
        return

    if not Path(watermark_path).exists():
        print(f"❌ Watermark image not found: {watermark_path}")
        print("Please run tests first to generate test images.")
        return

    # Create evaluator and run evaluation
    evaluator = AttackEvaluator(host_path, watermark_path)
    results = evaluator.run_evaluation()

    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved in: {evaluator.output_dir}")


if __name__ == "__main__":
    main()