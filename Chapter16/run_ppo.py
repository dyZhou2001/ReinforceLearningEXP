#!/usr/bin/env python3
"""
PPO训练启动脚本
提供简单的命令行接口来启动PPO训练
"""

import subprocess
import sys
import os

def main():
    print("PPO强化学习训练启动器")
    print("=" * 40)
    
    # 检查Python环境
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 提供训练选项
    print("\n选择训练配置:")
    print("1. 快速测试训练 (100 episodes)")
    print("2. 标准训练 (1000 episodes)")
    print("3. 长时间训练 (5000 episodes)")
    print("4. 自定义训练")
    print("5. 测试已有模型")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 快速测试训练
        cmd = [
            sys.executable, "PPOALLINONE.py",
            "--max_episodes", "100",
            "--save_dir", "./ppo_saves_quick",
            "--log_dir", "./ppo_logs_quick"
        ]
        print("\n启动快速测试训练...")
        
    elif choice == "2":
        # 标准训练
        cmd = [
            sys.executable, "PPOALLINONE.py",
            "--max_episodes", "1000",
            "--save_dir", "./ppo_saves",
            "--log_dir", "./ppo_logs"
        ]
        print("\n启动标准训练...")
        
    elif choice == "3":
        # 长时间训练
        cmd = [
            sys.executable, "PPOALLINONE.py",
            "--max_episodes", "5000",
            "--save_dir", "./ppo_saves_long",
            "--log_dir", "./ppo_logs_long"
        ]
        print("\n启动长时间训练...")
        
    elif choice == "4":
        # 自定义训练
        episodes = input("输入训练episodes数 (默认1000): ").strip()
        episodes = episodes if episodes else "1000"
        
        device = input("选择设备 cuda/cpu (默认cuda): ").strip()
        device = device if device else "cuda"
        
        cmd = [
            sys.executable, "PPOALLINONE.py",
            "--max_episodes", episodes,
            "--device", device,
            "--save_dir", f"./ppo_saves_custom_{episodes}",
            "--log_dir", f"./ppo_logs_custom_{episodes}"
        ]
        print(f"\n启动自定义训练 ({episodes} episodes, {device})...")
        
    elif choice == "5":
        # 测试模型
        model_path = input("输入模型路径 (默认./ppo_saves/best_model.pth): ").strip()
        model_path = model_path if model_path else "./ppo_saves/best_model.pth"
        
        episodes = input("输入测试episodes数 (默认10): ").strip()
        episodes = episodes if episodes else "10"
        
        render = input("是否渲染环境? y/n (默认n): ").strip().lower()
        
        cmd = [sys.executable, "test_ppo.py", "--model_path", model_path, "--episodes", episodes]
        if render == "y":
            cmd.append("--render")
        
        print(f"\n开始测试模型...")
        
    else:
        print("无效选择，退出。")
        return
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print("\n完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n执行过程中出现错误: {e}")
    except KeyboardInterrupt:
        print("\n用户中断训练。")
    except Exception as e:
        print(f"\n未知错误: {e}")


if __name__ == "__main__":
    main()