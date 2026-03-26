#!/usr/bin/env python3
"""
Performance Benchmark: Sequential vs Parallel Document Processing
Compares processing speed with different worker counts
"""
import time
import os
from pathlib import Path
from parallel_ingest import ParallelDocumentProcessor


def benchmark_ingestion(directory="./documents", max_files=None):
    """Benchmark parallel processing with different worker counts

    Args:
        directory: Path to documents
        max_files: Optional limit for testing (None = all files)
    """
    print("\n" + "="*80)
    print("PARALLEL PROCESSING BENCHMARK")
    print("="*80 + "\n")

    # Count available files
    pdf_files = list(Path(directory).glob("**/*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    total_files = len(pdf_files)
    print(f"📊 Files to process: {total_files}")
    print(f"📁 Directory: {directory}\n")

    if total_files == 0:
        print("✗ No PDF files found. Exiting.")
        return

    # Test configurations
    worker_configs = [1, 2, 4, 8]

    results = []

    for workers in worker_configs:
        print(f"\n{'─'*80}")
        print(f"Testing with {workers} worker(s)...")
        print(f"{'─'*80}")

        # Initialize processor
        processor = ParallelDocumentProcessor(
            db_path="./legal_audit_db_benchmark",  # Separate DB for testing
            max_workers=workers
        )

        # Measure processing time
        start_time = time.time()

        result = processor.ingest_directory_parallel(directory)

        elapsed_time = time.time() - start_time

        # Store results
        if result['success']:
            results.append({
                'workers': workers,
                'time': elapsed_time,
                'files': result['successful'],
                'chunks': result['total_chunks'],
                'failed': result['failed']
            })

            print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
            print(f"  Files processed: {result['successful']}/{result['total_files']}")
            print(f"  Total chunks: {result['total_chunks']}")

    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80 + "\n")

    if results:
        baseline_time = results[0]['time']  # 1 worker = sequential baseline

        print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Files/sec':<12}")
        print("─" * 48)

        for r in results:
            speedup = baseline_time / r['time']
            files_per_sec = r['files'] / r['time']

            print(f"{r['workers']:<10} {r['time']:<12.2f} "
                  f"{speedup:<12.2f}x {files_per_sec:<12.2f}")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        best_config = max(results, key=lambda x: x['time'] and (results[0]['time'] / x['time']))

        print(f"\n✓ Optimal configuration: {best_config['workers']} workers")
        print(f"  Speedup: {baseline_time / best_config['time']:.2f}x faster than sequential")
        print(f"  Processing time: {best_config['time']:.2f}s for {best_config['files']} files")
        print(f"  Throughput: {best_config['files'] / best_config['time']:.2f} files/second")

        # Extrapolate to full 220 documents
        if total_files < 220:
            estimated_time_full = (220 / total_files) * best_config['time']
            estimated_time_sequential = (220 / total_files) * baseline_time

            print(f"\n📊 Estimated time for 220 documents:")
            print(f"  Sequential (1 worker): {estimated_time_sequential/60:.1f} minutes")
            print(f"  Parallel ({best_config['workers']} workers): {estimated_time_full/60:.1f} minutes")
            print(f"  Time saved: {(estimated_time_sequential - estimated_time_full)/60:.1f} minutes")

    # Cleanup
    print(f"\n{'─'*80}")
    print("Cleaning up benchmark database...")
    os.system("rm -rf ./legal_audit_db_benchmark")
    print("✓ Benchmark complete")
    print("="*80 + "\n")


def quick_test(sample_size=10):
    """Quick test with small sample of files

    Args:
        sample_size: Number of files to test with
    """
    print(f"\n🚀 Quick Test Mode: {sample_size} files\n")
    benchmark_ingestion("./documents", max_files=sample_size)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Quick test mode
        sample = int(sys.argv[1]) if sys.argv[1].isdigit() else 10
        quick_test(sample)
    else:
        # Full benchmark (or limited to available files)
        benchmark_ingestion("./documents")
