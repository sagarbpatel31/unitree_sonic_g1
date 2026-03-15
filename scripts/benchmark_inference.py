#!/usr/bin/env python3
"""
Inference performance benchmarking tool for deployed G1 controllers.

This script provides comprehensive performance testing of exported models
across different hardware configurations and inference engines.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sonic_g1.deployment.inference import RuntimeInferenceEngine, ONNXInferenceEngine, TorchScriptInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """
    Comprehensive inference performance benchmarking suite.

    Tests inference latency, throughput, memory usage, and accuracy
    across different model formats and hardware configurations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results = {}

        # Benchmark parameters
        self.warmup_iterations = config.get('warmup_iterations', 100)
        self.benchmark_iterations = config.get('benchmark_iterations', 1000)
        self.batch_sizes = config.get('batch_sizes', [1])
        self.input_shapes = config.get('input_shapes', [(150,)])  # G1 observation size

        logger.info("Initialized InferenceBenchmark")

    def benchmark_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """
        Benchmark a single model across different configurations.

        Args:
            model_path: Path to model file
            model_name: Name for identification

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking model: {model_name} ({model_path})")

        model_results = {
            'model_name': model_name,
            'model_path': model_path,
            'model_size_mb': self._get_model_size(model_path),
            'benchmarks': {}
        }

        # Test different inference engines
        engines_to_test = self._get_engines_to_test(model_path)

        for engine_name, engine in engines_to_test.items():
            if engine is None:
                continue

            logger.info(f"Testing engine: {engine_name}")

            try:
                engine_results = self._benchmark_engine(engine, engine_name)
                model_results['benchmarks'][engine_name] = engine_results

            except Exception as e:
                logger.error(f"Failed to benchmark {engine_name}: {e}")
                model_results['benchmarks'][engine_name] = {
                    'error': str(e)
                }

        return model_results

    def _get_engines_to_test(self, model_path: str) -> Dict[str, Any]:
        """Get list of inference engines to test for the model."""
        engines = {}

        # Determine model format
        model_path_obj = Path(model_path)
        extension = model_path_obj.suffix.lower()

        if extension == '.onnx':
            try:
                # Test CPU provider
                engines['onnx_cpu'] = ONNXInferenceEngine(
                    model_path, providers=['CPUExecutionProvider']
                )

                # Test CUDA provider if available
                try:
                    engines['onnx_cuda'] = ONNXInferenceEngine(
                        model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                except:
                    logger.info("CUDA provider not available for ONNX")

            except Exception as e:
                logger.error(f"Failed to create ONNX engines: {e}")

        elif extension in ['.pt', '.pth']:
            try:
                # Test CPU
                engines['torchscript_cpu'] = TorchScriptInferenceEngine(
                    model_path, device='cpu'
                )

                # Test CUDA if available
                try:
                    engines['torchscript_cuda'] = TorchScriptInferenceEngine(
                        model_path, device='cuda'
                    )
                except:
                    logger.info("CUDA not available for TorchScript")

            except Exception as e:
                logger.error(f"Failed to create TorchScript engines: {e}")

        return engines

    def _benchmark_engine(self, engine: Any, engine_name: str) -> Dict[str, Any]:
        """Benchmark a specific inference engine."""
        results = {
            'engine_name': engine_name,
            'latency': {},
            'throughput': {},
            'memory': {},
            'accuracy': {}
        }

        # Test different input configurations
        for batch_size in self.batch_sizes:
            for input_shape in self.input_shapes:
                config_name = f"batch_{batch_size}_shape_{input_shape}"
                logger.info(f"Testing {engine_name} with {config_name}")

                # Create test input
                full_shape = (batch_size,) + input_shape
                test_input = np.random.randn(*full_shape).astype(np.float32)

                # Warmup
                self._warmup_engine(engine, test_input)

                # Latency benchmark
                latency_results = self._benchmark_latency(engine, test_input)
                results['latency'][config_name] = latency_results

                # Throughput benchmark
                throughput_results = self._benchmark_throughput(engine, test_input)
                results['throughput'][config_name] = throughput_results

                # Memory usage (if possible to measure)
                memory_results = self._benchmark_memory_usage(engine, test_input)
                results['memory'][config_name] = memory_results

        # Accuracy benchmark (determinism check)
        accuracy_results = self._benchmark_accuracy(engine)
        results['accuracy'] = accuracy_results

        return results

    def _warmup_engine(self, engine: Any, test_input: np.ndarray):
        """Warmup inference engine."""
        logger.debug(f"Warming up engine with {self.warmup_iterations} iterations")

        for _ in range(self.warmup_iterations):
            try:
                _ = engine.predict(test_input)
            except Exception as e:
                logger.error(f"Warmup failed: {e}")
                break

    def _benchmark_latency(self, engine: Any, test_input: np.ndarray) -> Dict[str, float]:
        """Benchmark inference latency."""
        latencies = []

        for _ in range(self.benchmark_iterations):
            start_time = time.time()
            try:
                _ = engine.predict(test_input)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            except Exception as e:
                logger.error(f"Latency benchmark failed: {e}")
                break

        if not latencies:
            return {'error': 'No successful inferences'}

        return {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'samples': len(latencies)
        }

    def _benchmark_throughput(self, engine: Any, test_input: np.ndarray) -> Dict[str, float]:
        """Benchmark inference throughput."""
        duration = 5.0  # seconds
        start_time = time.time()
        iterations = 0

        while (time.time() - start_time) < duration:
            try:
                _ = engine.predict(test_input)
                iterations += 1
            except Exception as e:
                logger.error(f"Throughput benchmark failed: {e}")
                break

        actual_duration = time.time() - start_time
        throughput = iterations / actual_duration

        return {
            'inferences_per_second': throughput,
            'total_iterations': iterations,
            'duration_seconds': actual_duration
        }

    def _benchmark_memory_usage(self, engine: Any, test_input: np.ndarray) -> Dict[str, Any]:
        """Benchmark memory usage (basic implementation)."""
        try:
            import psutil
            import os

            # Get process info
            process = psutil.Process(os.getpid())

            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run some inferences
            for _ in range(100):
                _ = engine.predict(test_input)

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            return {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_after - memory_before
            }

        except ImportError:
            return {'error': 'psutil not available for memory monitoring'}
        except Exception as e:
            return {'error': f'Memory benchmark failed: {e}'}

    def _benchmark_accuracy(self, engine: Any) -> Dict[str, Any]:
        """Benchmark output accuracy/determinism."""
        # Test determinism - same input should give same output
        test_input = np.random.randn(1, self.input_shapes[0][0]).astype(np.float32)

        outputs = []
        for _ in range(10):
            try:
                output = engine.predict(test_input)
                outputs.append(output)
            except Exception as e:
                return {'error': f'Accuracy benchmark failed: {e}'}

        if len(outputs) < 2:
            return {'error': 'Not enough outputs for comparison'}

        # Check consistency
        reference_output = outputs[0]
        differences = []

        for output in outputs[1:]:
            if output.shape != reference_output.shape:
                return {'error': 'Inconsistent output shapes'}

            diff = np.abs(output - reference_output)
            differences.append(np.max(diff))

        return {
            'max_difference': max(differences),
            'mean_difference': np.mean(differences),
            'deterministic': max(differences) < 1e-6,
            'output_shape': list(reference_output.shape)
        }

    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB."""
        try:
            size_bytes = Path(model_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0.0

    def benchmark_runtime_engine(self, config_path: str) -> Dict[str, Any]:
        """Benchmark the complete runtime inference engine."""
        logger.info("Benchmarking RuntimeInferenceEngine")

        # Load configuration
        with open(config_path, 'r') as f:
            runtime_config = json.load(f)

        try:
            # Create runtime engine
            runtime_engine = RuntimeInferenceEngine(runtime_config)

            # Create test observation
            obs_dim = 150  # G1 observation dimension
            test_obs = np.random.randn(obs_dim).astype(np.float32)

            # Benchmark with full pipeline
            latencies = []
            safety_checks = []

            for _ in range(self.benchmark_iterations):
                start_time = time.time()

                action, info = runtime_engine.predict(test_obs, enable_safety=True)

                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)

                # Track safety features
                safety_checks.append({
                    'timeout': info.get('timeout', False),
                    'inference_time_ms': info.get('inference_time_ms', 0)
                })

            # Collect statistics
            stats = runtime_engine.get_statistics()

            return {
                'runtime_engine': {
                    'latency': {
                        'mean_ms': np.mean(latencies),
                        'std_ms': np.std(latencies),
                        'p95_ms': np.percentile(latencies, 95),
                        'p99_ms': np.percentile(latencies, 99)
                    },
                    'safety': {
                        'timeout_rate': np.mean([s['timeout'] for s in safety_checks]),
                        'avg_inference_time_ms': np.mean([s['inference_time_ms'] for s in safety_checks])
                    },
                    'engine_stats': stats
                }
            }

        except Exception as e:
            logger.error(f"Runtime engine benchmark failed: {e}")
            return {'runtime_engine': {'error': str(e)}}

    def run_comprehensive_benchmark(self, models: List[Dict[str, str]],
                                   output_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple models.

        Args:
            models: List of model configs with 'name' and 'path' keys
            output_dir: Directory to save results

        Returns:
            Complete benchmark results
        """
        logger.info(f"Running comprehensive benchmark on {len(models)} models")

        results = {
            'benchmark_config': self.config,
            'timestamp': time.time(),
            'models': {},
            'summary': {}
        }

        # Benchmark each model
        for model_config in models:
            model_name = model_config['name']
            model_path = model_config['path']

            if not Path(model_path).exists():
                logger.error(f"Model not found: {model_path}")
                continue

            model_results = self.benchmark_model(model_path, model_name)
            results['models'][model_name] = model_results

        # Generate summary
        results['summary'] = self._generate_summary(results['models'])

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate plots
        self._generate_plots(results, output_path)

        # Generate report
        self._generate_report(results, output_path / "benchmark_report.txt")

        logger.info(f"Benchmark completed. Results saved to {output_dir}")
        return results

    def _generate_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all models."""
        summary = {
            'total_models': len(model_results),
            'engines_tested': set(),
            'best_latency': {},
            'best_throughput': {},
            'model_comparison': {}
        }

        all_latencies = {}
        all_throughputs = {}

        for model_name, model_data in model_results.items():
            if 'benchmarks' not in model_data:
                continue

            for engine_name, engine_data in model_data['benchmarks'].items():
                if 'error' in engine_data:
                    continue

                summary['engines_tested'].add(engine_name)

                # Collect latency data
                if 'latency' in engine_data:
                    for config, latency_data in engine_data['latency'].items():
                        if 'mean_ms' in latency_data:
                            key = f"{engine_name}_{config}"
                            all_latencies[key] = latency_data['mean_ms']

                # Collect throughput data
                if 'throughput' in engine_data:
                    for config, throughput_data in engine_data['throughput'].items():
                        if 'inferences_per_second' in throughput_data:
                            key = f"{engine_name}_{config}"
                            all_throughputs[key] = throughput_data['inferences_per_second']

        # Find best performers
        if all_latencies:
            best_latency_key = min(all_latencies.keys(), key=lambda k: all_latencies[k])
            summary['best_latency'] = {
                'config': best_latency_key,
                'latency_ms': all_latencies[best_latency_key]
            }

        if all_throughputs:
            best_throughput_key = max(all_throughputs.keys(), key=lambda k: all_throughputs[k])
            summary['best_throughput'] = {
                'config': best_throughput_key,
                'throughput_ips': all_throughputs[best_throughput_key]
            }

        summary['engines_tested'] = list(summary['engines_tested'])
        return summary

    def _generate_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate benchmark visualization plots."""
        try:
            # Latency comparison plot
            self._plot_latency_comparison(results, output_dir / "latency_comparison.png")

            # Throughput comparison plot
            self._plot_throughput_comparison(results, output_dir / "throughput_comparison.png")

            # Model size vs performance
            self._plot_size_vs_performance(results, output_dir / "size_vs_performance.png")

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

    def _plot_latency_comparison(self, results: Dict[str, Any], save_path: Path):
        """Plot latency comparison across engines and models."""
        plt.figure(figsize=(12, 8))

        data = []
        for model_name, model_data in results['models'].items():
            if 'benchmarks' not in model_data:
                continue

            for engine_name, engine_data in model_data['benchmarks'].items():
                if 'latency' in engine_data:
                    for config, latency_data in engine_data['latency'].items():
                        if 'mean_ms' in latency_data:
                            data.append({
                                'Model': model_name,
                                'Engine': engine_name,
                                'Config': config,
                                'Latency (ms)': latency_data['mean_ms']
                            })

        if data:
            df = pd.DataFrame(data)

            # Create grouped bar plot
            engines = df['Engine'].unique()
            models = df['Model'].unique()

            x = np.arange(len(models))
            width = 0.8 / len(engines)

            for i, engine in enumerate(engines):
                engine_data = df[df['Engine'] == engine]
                latencies = [engine_data[engine_data['Model'] == model]['Latency (ms)'].mean()
                           if len(engine_data[engine_data['Model'] == model]) > 0 else 0
                           for model in models]

                plt.bar(x + i * width, latencies, width, label=engine, alpha=0.8)

            plt.xlabel('Model')
            plt.ylabel('Average Latency (ms)')
            plt.title('Inference Latency Comparison')
            plt.xticks(x + width * (len(engines) - 1) / 2, models, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_throughput_comparison(self, results: Dict[str, Any], save_path: Path):
        """Plot throughput comparison."""
        plt.figure(figsize=(12, 8))

        data = []
        for model_name, model_data in results['models'].items():
            if 'benchmarks' not in model_data:
                continue

            for engine_name, engine_data in model_data['benchmarks'].items():
                if 'throughput' in engine_data:
                    for config, throughput_data in engine_data['throughput'].items():
                        if 'inferences_per_second' in throughput_data:
                            data.append({
                                'Model': model_name,
                                'Engine': engine_name,
                                'Throughput (IPS)': throughput_data['inferences_per_second']
                            })

        if data:
            df = pd.DataFrame(data)

            engines = df['Engine'].unique()
            models = df['Model'].unique()

            x = np.arange(len(models))
            width = 0.8 / len(engines)

            for i, engine in enumerate(engines):
                engine_data = df[df['Engine'] == engine]
                throughputs = [engine_data[engine_data['Model'] == model]['Throughput (IPS)'].mean()
                             if len(engine_data[engine_data['Model'] == model]) > 0 else 0
                             for model in models]

                plt.bar(x + i * width, throughputs, width, label=engine, alpha=0.8)

            plt.xlabel('Model')
            plt.ylabel('Throughput (Inferences/Second)')
            plt.title('Inference Throughput Comparison')
            plt.xticks(x + width * (len(engines) - 1) / 2, models, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_size_vs_performance(self, results: Dict[str, Any], save_path: Path):
        """Plot model size vs performance."""
        plt.figure(figsize=(10, 6))

        for model_name, model_data in results['models'].items():
            if 'benchmarks' not in model_data or 'model_size_mb' not in model_data:
                continue

            size_mb = model_data['model_size_mb']

            for engine_name, engine_data in model_data['benchmarks'].items():
                if 'latency' in engine_data:
                    # Get average latency across all configs
                    latencies = []
                    for config, latency_data in engine_data['latency'].items():
                        if 'mean_ms' in latency_data:
                            latencies.append(latency_data['mean_ms'])

                    if latencies:
                        avg_latency = np.mean(latencies)
                        plt.scatter(size_mb, avg_latency, label=f"{model_name} ({engine_name})", s=100, alpha=0.7)

        plt.xlabel('Model Size (MB)')
        plt.ylabel('Average Latency (ms)')
        plt.title('Model Size vs Inference Latency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, results: Dict[str, Any], report_path: Path):
        """Generate text report summary."""
        with open(report_path, 'w') as f:
            f.write("INFERENCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Summary
            summary = results.get('summary', {})
            f.write(f"Total Models Tested: {summary.get('total_models', 0)}\n")
            f.write(f"Engines Tested: {', '.join(summary.get('engines_tested', []))}\n")

            if 'best_latency' in summary:
                f.write(f"Best Latency: {summary['best_latency']['latency_ms']:.2f}ms ({summary['best_latency']['config']})\n")

            if 'best_throughput' in summary:
                f.write(f"Best Throughput: {summary['best_throughput']['throughput_ips']:.1f} IPS ({summary['best_throughput']['config']})\n")

            f.write("\n" + "=" * 50 + "\n\n")

            # Detailed results
            for model_name, model_data in results['models'].items():
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Size: {model_data.get('model_size_mb', 0):.1f} MB\n")

                if 'benchmarks' in model_data:
                    for engine_name, engine_data in model_data['benchmarks'].items():
                        f.write(f"\n{engine_name}:\n")

                        if 'error' in engine_data:
                            f.write(f"  Error: {engine_data['error']}\n")
                            continue

                        if 'latency' in engine_data:
                            for config, latency_data in engine_data['latency'].items():
                                if 'mean_ms' in latency_data:
                                    f.write(f"  {config}: {latency_data['mean_ms']:.2f}ms (±{latency_data['std_ms']:.2f})\n")

                f.write("\n")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark G1 inference performance")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="Paths to model files to benchmark")
    parser.add_argument("--names", type=str, nargs='+',
                       help="Names for models (optional)")
    parser.add_argument("--runtime_config", type=str,
                       help="Path to runtime engine config for testing")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="Number of benchmark iterations")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1],
                       help="Batch sizes to test")

    args = parser.parse_args()

    # Create benchmark configuration
    config = {
        'warmup_iterations': 100,
        'benchmark_iterations': args.iterations,
        'batch_sizes': args.batch_sizes,
        'input_shapes': [(150,)]  # G1 observation dimension
    }

    # Prepare model list
    models = []
    for i, model_path in enumerate(args.models):
        name = args.names[i] if args.names and i < len(args.names) else f"Model_{i+1}"
        models.append({'name': name, 'path': model_path})

    # Run benchmark
    benchmark = InferenceBenchmark(config)
    results = benchmark.run_comprehensive_benchmark(models, args.output_dir)

    # Test runtime engine if config provided
    if args.runtime_config:
        runtime_results = benchmark.benchmark_runtime_engine(args.runtime_config)
        results['runtime_engine'] = runtime_results

        # Save updated results
        output_path = Path(args.output_dir)
        results_file = output_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    summary = results.get('summary', {})
    print(f"\nBenchmark Summary:")
    print(f"Models tested: {summary.get('total_models', 0)}")
    print(f"Engines: {', '.join(summary.get('engines_tested', []))}")

    if 'best_latency' in summary:
        print(f"Best latency: {summary['best_latency']['latency_ms']:.2f}ms")

    if 'best_throughput' in summary:
        print(f"Best throughput: {summary['best_throughput']['throughput_ips']:.1f} IPS")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()