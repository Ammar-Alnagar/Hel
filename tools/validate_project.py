#!/usr/bin/env python3
"""
Project validation script for Helios Engine.

This script checks for common issues in the project structure,
code quality, and documentation completeness.
"""

import os
import re
import sys
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()


def check_cmakelists():
    """Validate CMakeLists.txt structure."""
    print("🔍 Checking CMakeLists.txt...")

    cmakelists = Path("CMakeLists.txt")
    if not cmakelists.exists():
        print("❌ CMakeLists.txt not found")
        return False

    content = cmakelists.read_text()

    # Check for required components
    required_patterns = [
        r'cmake_minimum_required',
        r'project\(',
        r'find_package\(Eigen3',
        r'find_package\(Protobuf',
        r'add_executable',
        r'target_link_libraries',
        r'src/',
    ]

    missing = []
    for pattern in required_patterns:
        if not re.search(pattern, content, re.IGNORECASE):
            missing.append(pattern)

    if missing:
        print(f"❌ Missing required CMake patterns: {missing}")
        return False

    print("✅ CMakeLists.txt looks good")
    return True


def check_source_structure():
    """Check that all required source files exist."""
    print("🔍 Checking source file structure...")

    required_files = [
        "src/tensor.hpp",
        "src/tensor.cpp",
        "src/alloc.hpp",
        "src/alloc.cpp",
        "src/main.cpp",
        "src/app.hpp",
        "src/app.cpp",
        "src/kernels/gemm_ref.hpp",
        "src/kernels/gemm_ref.cpp",
        "src/kernels/q4_rowwise.hpp",
        "src/kernels/q4_rowwise.cpp",
        "src/loaders/onnx_loader.hpp",
        "src/loaders/onnx_loader.cpp",
        "src/transformer/transformer.hpp",
        "src/transformer/transformer.cpp",
        "src/tokenizer/sentencepiece_wrapper.hpp",
        "src/tokenizer/sentencepiece_wrapper.cpp",
        "src/util/threadpool.hpp",
        "src/util/threadpool.cpp",
        "src/util/profiler.hpp",
        "src/util/profiler.cpp",
    ]

    missing_files = []
    for filepath in required_files:
        if not check_file_exists(filepath):
            missing_files.append(filepath)

    if missing_files:
        print(f"❌ Missing source files: {missing_files}")
        return False

    print("✅ All required source files present")
    return True


def check_headers():
    """Check that header files have proper include guards."""
    print("🔍 Checking header include guards...")

    header_files = [
        "src/tensor.hpp",
        "src/alloc.hpp",
        "src/app.hpp",
        "src/kernels/gemm_ref.hpp",
        "src/kernels/q4_rowwise.hpp",
        "src/loaders/onnx_loader.hpp",
        "src/transformer/transformer.hpp",
        "src/tokenizer/sentencepiece_wrapper.hpp",
        "src/util/threadpool.hpp",
        "src/util/profiler.hpp",
    ]

    issues = []
    for header in header_files:
        if not check_file_exists(header):
            continue

        content = Path(header).read_text()

        # Extract filename for guard check
        filename = Path(header).stem.upper()

        # Check for include guard pattern
        guard_pattern = f"#ifndef {filename}_HPP|#ifndef {filename}_H"
        pragma_pattern = "#pragma once"

        if not re.search(guard_pattern, content) and not re.search(pragma_pattern, content):
            issues.append(f"{header}: Missing include guard")

    if issues:
        print(f"❌ Header guard issues: {issues}")
        return False

    print("✅ All headers have proper include guards")
    return True


def check_documentation():
    """Check that key documentation files exist and are complete."""
    print("🔍 Checking documentation...")

    # Check README
    readme = Path("README.md")
    if not readme.exists():
        print("❌ README.md not found")
        return False

    content = readme.read_text()

    # Check for key sections (with flexible matching)
    required_sections = [
        "🚀 Helios Engine - High-Performance LLM Inference",
        "## 🌟 Overview",
        "## 🏗️ Architecture",
        "## 🚀 Quick Start",
        "## 📊 Performance",
    ]

    missing_sections = []
    for section in required_sections:
        # Use word boundary matching for more flexible detection
        if section not in content:
            missing_sections.append(section)

    if missing_sections:
        print(f"❌ Missing README sections: {missing_sections}")
        return False

    # Check LICENSE
    license_file = Path("LICENSE")
    if not license_file.exists():
        print("❌ LICENSE file not found")
        return False

    print("✅ Documentation looks complete")
    return True


def check_tools():
    """Check that utility tools are present."""
    print("🔍 Checking utility tools...")

    tools = ["tools/convert_model.py", "tools/benchmark.py", "tools/test_manual.py"]

    missing_tools = []
    for tool in tools:
        if not check_file_exists(tool):
            missing_tools.append(tool)

    if missing_tools:
        print(f"❌ Missing utility tools: {missing_tools}")
        return False

    print("✅ All utility tools present")
    return True


def main():
    """Run all validation checks."""
    print("🚀 Helios Engine Project Validation")
    print("=" * 50)

    checks = [
        ("CMake Configuration", check_cmakelists),
        ("Source Structure", check_source_structure),
        ("Header Guards", check_headers),
        ("Documentation", check_documentation),
        ("Utility Tools", check_tools),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}: Error - {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")

    passed = 0
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if result:
            passed += 1

    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 Project validation successful!")
        return 0
    else:
        print("❌ Some validation checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
