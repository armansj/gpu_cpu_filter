import torch
import time
import matplotlib.pyplot as plt


def find_divisors_cpu(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def find_divisors_gpu(n, device):
    numbers = torch.arange(1, n + 1, device=device)

    mask = n % numbers == 0
    divisors = numbers[mask]

    return divisors


def compare_cpu_gpu_performance(nums):
    results = []
    cpu_times = []
    gpu_times = []

    for n in nums:
        print(f"Testing for n = {n}")

        start_time_cpu = time.time()
        divisors_cpu = find_divisors_cpu(n)
        end_time_cpu = time.time()
        cpu_time = end_time_cpu - start_time_cpu

        print(f"CPU Time for n={n}: {cpu_time:.4f} seconds")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        start_time_gpu = time.time()
        divisors_gpu = find_divisors_gpu(n, device)
        end_time_gpu = time.time()
        gpu_time = end_time_gpu - start_time_gpu

        print(f"MPS GPU Time for n={n}: {gpu_time:.4f} seconds")

        results.append({
            "n": n,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time
        })

        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

    return results, cpu_times, gpu_times


nums_to_test = [1000000, 5000000, 10000000, 15000000,50000000,75000000,100000000]

results, cpu_times, gpu_times = compare_cpu_gpu_performance(nums_to_test)

print("\nSummary of CPU vs GPU performance:")
for result in results:
    print(f"n={result['n']}: CPU Time = {result['cpu_time']:.4f}s, GPU Time = {result['gpu_time']:.4f}s")

plt.figure(figsize=(10, 6))
plt.plot(nums_to_test, cpu_times, label="CPU Time", color="blue", marker='o')
plt.plot(nums_to_test, gpu_times, label="GPU Time", color="red", marker='x')
plt.xlabel('Number (n)')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Performance for Finding Divisors')
plt.legend()
plt.grid(True)
plt.show()
