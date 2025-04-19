import subprocess
import os
import shutil

def setup_emscripten():
    """
    Check if emscripten is installed and set up environment
    """
    try:
        subprocess.run(['emcc', '--version'], check=True, capture_output=True)
        print("Emscripten is already installed")
    except:
        print("Installing emscripten...")
        subprocess.run(['git', 'clone', 'https://github.com/emscripten-core/emsdk.git'], check=True)
        os.chdir('emsdk')
        subprocess.run(['./emsdk', 'install', 'latest'], check=True)
        subprocess.run(['./emsdk', 'activate', 'latest'], check=True)
        subprocess.run(['source', './emsdk_env.sh'], check=True)
        os.chdir('..')

def setup_build_env():
    """
    Set up the build environment
    """
    # Create build directory
    os.makedirs('build', exist_ok=True)
    
    # Copy required files to the build directory
    if os.path.exists('model'):
        shutil.copytree('model', 'build/model', dirs_exist_ok=True)
    if os.path.exists('kinetics400.csv'):
        shutil.copy2('kinetics400.csv', 'build/')

def build_wasm():
    """
    Build the WebAssembly module using CMake and Emscripten
    """
    print("Setting up build environment...")
    setup_build_env()
    
    print("Configuring CMake...")
    cmake_config_cmd = [
        'emcmake', 'cmake',
        '-B', 'build',
        '-DCMAKE_BUILD_TYPE=Release',
        '.'
    ]
    
    try:
        subprocess.run(cmake_config_cmd, check=True)
        
        print("Building WebAssembly module...")
        cmake_build_cmd = ['cmake', '--build', 'build', '--config', 'Release']
        subprocess.run(cmake_build_cmd, check=True)
        
        # Copy output files to js directory
        os.makedirs('js', exist_ok=True)
        if os.path.exists('build/videomae_wasm.js'):
            shutil.copy2('build/videomae_wasm.js', 'js/')
        if os.path.exists('build/videomae_wasm.wasm'):
            shutil.copy2('build/videomae_wasm.wasm', 'js/')
        if os.path.exists('build/videomae_wasm.data'):
            shutil.copy2('build/videomae_wasm.data', 'js/')
        
        print("WebAssembly build completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        raise

if __name__ == '__main__':
    setup_emscripten()
    build_wasm()