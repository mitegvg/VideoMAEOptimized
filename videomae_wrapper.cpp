#include <emscripten.h>
#include <emscripten/bind.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

class VideoMAEInference {
private:
    torch::jit::script::Module model;
    std::vector<std::string> class_names;

    void load_class_names() {
        std::ifstream file("kinetics400.csv");
        std::string line;
        while (std::getline(file, line)) {
            size_t pos = line.find(',');
            if (pos != std::string::npos) {
                class_names.push_back(line.substr(pos + 1));
            }
        }
    }

public:
    VideoMAEInference() {
        try {
            // Load the TorchScript model
            model = torch::jit::load("model/pytorch_model.bin");
            model.eval();
            load_class_names();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<std::pair<std::string, float>> run_inference(
        const std::vector<float>& frames_data,
        int num_frames,
        int height = 224,
        int width = 224
    ) {
        try {
            // Reshape input data to [1, num_frames, 3, height, width]
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto input_tensor = torch::from_blob(
                const_cast<float*>(frames_data.data()),
                {1, num_frames, 3, height, width},
                options
            );

            // Normalize input (assuming frames are in range [0, 1])
            input_tensor = input_tensor.sub_(0.5).div_(0.5);

            // Create inputs vector
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // Run inference
            auto output = model.forward(inputs).toTensor();
            
            // Get top 5 predictions
            auto softmax_output = torch::softmax(output, 1);
            auto topk = torch::topk(softmax_output, 5);
            auto probs = std::get<0>(topk).squeeze();
            auto indices = std::get<1>(topk).squeeze();

            // Convert to result format
            std::vector<std::pair<std::string, float>> results;
            for (int i = 0; i < 5; i++) {
                int idx = indices[i].item<int>();
                float prob = probs[i].item<float>();
                results.emplace_back(class_names[idx], prob);
            }

            return results;
        } catch (const c10::Error& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            throw;
        }
    }
};

// Wrapper functions for JavaScript
extern "C" {
    EMSCRIPTEN_KEEPALIVE VideoMAEInference* create_inference() {
        return new VideoMAEInference();
    }

    EMSCRIPTEN_KEEPALIVE void destroy_inference(VideoMAEInference* inference) {
        delete inference;
    }

    EMSCRIPTEN_KEEPALIVE void* run_inference(
        VideoMAEInference* inference,
        float* frames_data,
        int num_frames
    ) {
        try {
            std::vector<float> frames_vec(frames_data, frames_data + (num_frames * 3 * 224 * 224));
            auto results = inference->run_inference(frames_vec, num_frames);
            
            // Allocate memory for results
            size_t result_size = results.size() * sizeof(struct {
                const char* class_name;
                float probability;
            });
            
            auto* result_ptr = static_cast<char*>(malloc(result_size));
            char* current = result_ptr;
            
            for (const auto& result : results) {
                // Copy class name
                size_t name_len = result.first.size() + 1;
                char* name_ptr = static_cast<char*>(malloc(name_len));
                strcpy(name_ptr, result.first.c_str());
                
                // Store pointer and probability
                *reinterpret_cast<const char**>(current) = name_ptr;
                current += sizeof(const char*);
                *reinterpret_cast<float*>(current) = result.second;
                current += sizeof(float);
            }
            
            return result_ptr;
        } catch (const std::exception& e) {
            std::cerr << "Error in run_inference: " << e.what() << std::endl;
            return nullptr;
        }
    }
}