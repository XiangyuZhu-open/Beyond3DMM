## Getting started
### Requirements
 - PyTorch >= 0.4.1 (**PyTorch v1.1.0** is tested successfully on macOS and Linux.)
 - Python >= 3.6 (Numpy, Scipy, Matplotlib)
 - Dlib (Dlib is optionally for face and landmarks detection. There is no need to use Dlib if you can provide face bouding bbox and landmarks. Besides, you can try the two-step inference strategy without initialized landmarks.)
 - OpenCV (Python version, for image IO operations.)
 - Cython (For accelerating depth and PNCC render.)
 - Platform: Linux or macOS (Windows is not tested.)

 ```
 # installation structions
 sudo pip3 install torch torchvision # for cpu version. more option to see https://pytorch.org
 sudo pip3 install numpy scipy matplotlib
 sudo pip3 install dlib==19.5.0 # 19.15+ version may cause conflict with pytorch in Linux, this may take several minutes. If 19.5 version raises errors, you may try 19.15+ version.
 sudo pip3 install opencv-python
 sudo pip3 install cython
 ```

In addition, I strongly recommend using Python3.6+ instead of older version for its better design.

### Usage

1. Clone this repo (this may take some time as it is a little big)
    ```
    ...

    ```
2. Build cython module (just one line for building)
   ```
   cd MM3D/cython
   python3 setup.py build_ext -i
   ```
   This is for accelerating depth estimation and PNCC render since Python is too slow in for loop.
   
    
3. Run the `main.py` with arbitrary image as input
    ```
    python3 main.py -f examples/images/test.jpg
    ```
    If you can see these output log in terminal, you run it successfully.
    ```
    The result of test are in examples/results
    ```
  