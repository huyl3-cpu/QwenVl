#!/usr/bin/env python3
"""
Check CPU and GPU specifications on Google Colab A100 80GB
"""

import subprocess
import platform
import psutil
import torch

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def get_cpu_info():
    """Get CPU information"""
    print_section("CPU INFORMATION")
    
    # Basic info
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores (threads): {psutil.cpu_count(logical=True)}")
    
    # CPU frequency
    freq = psutil.cpu_freq()
    if freq:
        print(f"Max frequency: {freq.max:.2f} MHz")
        print(f"Current frequency: {freq.current:.2f} MHz")
    
    # CPU model from /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
            model_name = [l for l in lines if 'model name' in l]
            if model_name:
                cpu_model = model_name[0].split(':')[1].strip()
                print(f"CPU Model: {cpu_model}")
    except:
        pass
    
    # CPU usage
    print(f"\nCPU Usage per core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"  Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent(interval=1)}%")

def get_memory_info():
    """Get RAM information"""
    print_section("MEMORY (RAM) INFORMATION")
    
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available: {mem.available / (1024**3):.2f} GB")
    print(f"Used: {mem.used / (1024**3):.2f} GB")
    print(f"Usage: {mem.percent}%")
    
    # Swap
    swap = psutil.swap_memory()
    print(f"\nSwap Total: {swap.total / (1024**3):.2f} GB")
    print(f"Swap Used: {swap.used / (1024**3):.2f} GB")

def get_gpu_info():
    """Get GPU information"""
    print_section("GPU INFORMATION")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")
            
            # Memory usage
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            mem_total = props.total_memory / (1024**3)
            
            print(f"  Memory allocated: {mem_allocated:.2f} GB")
            print(f"  Memory reserved: {mem_reserved:.2f} GB")
            print(f"  Memory free: {mem_total - mem_reserved:.2f} GB")
            print(f"  Utilization: {(mem_reserved / mem_total * 100):.1f}%")
    else:
        print("CUDA not available")

def get_nvidia_smi():
    """Get nvidia-smi output"""
    print_section("NVIDIA-SMI OUTPUT")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
    except FileNotFoundError:
        print("nvidia-smi not found")

def get_disk_info():
    """Get disk information"""
    print_section("DISK INFORMATION")
    
    disk = psutil.disk_usage('/')
    print(f"Total disk space: {disk.total / (1024**3):.2f} GB")
    print(f"Used: {disk.used / (1024**3):.2f} GB")
    print(f"Free: {disk.free / (1024**3):.2f} GB")
    print(f"Usage: {disk.percent}%")

def get_system_info():
    """Get general system information"""
    print_section("SYSTEM INFORMATION")
    
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Python version: {platform.python_version()}")
    
    # Uptime
    boot_time = psutil.boot_time()
    import datetime
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(boot_time)
    print(f"System uptime: {uptime}")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("  GOOGLE COLAB A100 SPECIFICATIONS")
    print("="*60)
    
    get_system_info()
    get_cpu_info()
    get_memory_info()
    get_gpu_info()
    get_disk_info()
    get_nvidia_smi()
    
    print("\n" + "="*60)
    print("  SUMMARY FOR OPTIMIZATION")
    print("="*60)
    
    # Quick summary
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"\n✓ CPU Cores: {cpu_count} threads")
    print(f"✓ RAM: {ram_gb:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        compute_cap = torch.cuda.get_device_properties(0).major
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ VRAM: {gpu_mem:.0f} GB")
        print(f"✓ Compute Capability: {compute_cap}.x")
        
        # Recommendations
        print("\n📊 RECOMMENDATIONS:")
        if compute_cap >= 8:
            print("  • BF16 supported ✓")
            print("  • Use quantization: BF16 for best speed")
        print(f"  • Available VRAM: {gpu_mem:.0f}GB → Use large models!")
        print(f"  • Suggested model: Qwen3-VL-8B or 32B")
    
    print()

if __name__ == "__main__":
    main()
