from setuptools import setup, find_packages

setup(
    name="EnetConvexHullPackage",
    version="0.1",
    description="A package for anomaly detection using Elastic Net and Convex Hull.",
    packages=find_packages(),  # پیدا کردن تمام پکیج‌های موجود
    install_requires=[
        "scikit-learn>=0.24.0",  # وابستگی‌های موردنیاز
        "numpy>=1.19.0",
        "qpsolvers",
        "tqdm",
    ],
    python_requires=">=3.7",  # حداقل نسخه پایتون موردنیاز
)
