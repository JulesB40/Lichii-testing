// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Builds a compute pipeline from each SPIR-V file given and prints the driver's result, so shaders
// a driver refuses (EDEN_NX2_SHADER_DUMP writes them as <hash>-<stage>.spv) and edited variants of
// them can be tried without the emulator. The descriptor set layouts come from the module's own
// bindings; every core feature the device supports is enabled, as the emulator does.
//
// Android: build with the NDK's clang (see spirv-pipeline-probe.md), push it and the .spv files to
// /data/local/tmp and run it from adb shell.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace {

std::vector<uint32_t> ReadWords(const char* path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) {
        return {};
    }
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> words(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return words;
}

struct Reflection {
    std::map<uint32_t, std::map<uint32_t, VkDescriptorSetLayoutBinding>> sets;
    bool push_constants{};
    std::string error;
};

Reflection Reflect(const std::vector<uint32_t>& words) {
    Reflection out;
    if (words.size() < 5 || words[0] != 0x07230203) {
        out.error = "not a SPIR-V module";
        return out;
    }
    struct Pointer {
        uint32_t storage;
        uint32_t type;
    };
    struct Image {
        uint32_t dim;
        uint32_t sampled;
    };
    std::map<uint32_t, Pointer> pointers;
    std::map<uint32_t, Image> images;
    std::map<uint32_t, uint32_t> sampled_images;
    std::set<uint32_t> samplers;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> arrays; // element, length id (0: runtime)
    std::map<uint32_t, uint32_t> constants;
    std::set<uint32_t> buffer_blocks;
    std::map<uint32_t, uint32_t> bindings;
    std::map<uint32_t, uint32_t> descriptor_sets;
    std::vector<std::pair<uint32_t, uint32_t>> variables; // id, pointer type
    for (size_t at = 5; at < words.size();) {
        const uint32_t count = words[at] >> 16;
        const uint32_t op = words[at] & 0xffff;
        if (count == 0 || at + count > words.size()) {
            out.error = "truncated module";
            return out;
        }
        const uint32_t* w = &words[at];
        switch (op) {
        case 25: // OpTypeImage
            images[w[1]] = {w[3], w[7]};
            break;
        case 26: // OpTypeSampler
            samplers.insert(w[1]);
            break;
        case 27: // OpTypeSampledImage
            sampled_images[w[1]] = w[2];
            break;
        case 28: // OpTypeArray
            arrays[w[1]] = {w[2], w[3]};
            break;
        case 29: // OpTypeRuntimeArray
            arrays[w[1]] = {w[2], 0};
            break;
        case 32: // OpTypePointer
            pointers[w[1]] = {w[2], w[3]};
            break;
        case 43: // OpConstant
            if (count >= 4) {
                constants[w[2]] = w[3];
            }
            break;
        case 59: // OpVariable
            variables.emplace_back(w[2], w[1]);
            break;
        case 71: // OpDecorate
            if (w[2] == 3) {
                buffer_blocks.insert(w[1]);
            } else if (w[2] == 33 && count >= 4) {
                bindings[w[1]] = w[3];
            } else if (w[2] == 34 && count >= 4) {
                descriptor_sets[w[1]] = w[3];
            }
            break;
        }
        at += count;
    }
    for (const auto& [id, pointer_type] : variables) {
        const auto pointer = pointers.find(pointer_type);
        if (pointer == pointers.end()) {
            continue;
        }
        const uint32_t storage = pointer->second.storage;
        if (storage == 9) { // PushConstant
            out.push_constants = true;
            continue;
        }
        const auto binding = bindings.find(id);
        if (binding == bindings.end()) {
            continue;
        }
        uint32_t type = pointer->second.type;
        uint32_t descriptors = 1;
        while (arrays.contains(type)) {
            const auto [element, length] = arrays[type];
            descriptors *= length != 0 && constants.contains(length) ? constants[length] : 64;
            type = element;
        }
        VkDescriptorType descriptor{};
        if (storage == 12) {
            descriptor = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } else if (storage == 2) {
            descriptor = buffer_blocks.contains(type) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                      : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (storage == 0) {
            if (samplers.contains(type)) {
                descriptor = VK_DESCRIPTOR_TYPE_SAMPLER;
            } else if (sampled_images.contains(type)) {
                const Image image = images[sampled_images[type]];
                descriptor = image.dim == 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            } else if (images.contains(type)) {
                const Image image = images[type];
                if (image.dim == 5) {
                    descriptor = image.sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                                                    : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                } else {
                    descriptor = image.sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                    : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                }
            } else {
                continue;
            }
        } else {
            continue;
        }
        const uint32_t set = descriptor_sets.contains(id) ? descriptor_sets[id] : 0;
        out.sets[set][binding->second] = VkDescriptorSetLayoutBinding{
            .binding = binding->second,
            .descriptorType = descriptor,
            .descriptorCount = descriptors,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        };
    }
    return out;
}

const char* ResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_INVALID_SHADER_NV:
        return "VK_ERROR_INVALID_SHADER_NV";
    case VK_ERROR_UNKNOWN:
        return "VK_ERROR_UNKNOWN";
    default:
        return "other";
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <module.spv>...\n", argv[0]);
        return 2;
    }
    const VkApplicationInfo app{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "spirv_pipeline_probe",
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_ci{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance instance{};
    if (vkCreateInstance(&instance_ci, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n");
        return 1;
    }
    uint32_t gpu_count = 1;
    VkPhysicalDevice gpu{};
    vkEnumeratePhysicalDevices(instance, &gpu_count, &gpu);
    if (gpu == VK_NULL_HANDLE) {
        std::fprintf(stderr, "no physical device\n");
        return 1;
    }
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &driver};
    vkGetPhysicalDeviceProperties2(gpu, &properties);
    std::printf("device %s, driver %s %s, api %u.%u.%u\n", properties.properties.deviceName,
                driver.driverName, driver.driverInfo, VK_API_VERSION_MAJOR(properties.properties.apiVersion),
                VK_API_VERSION_MINOR(properties.properties.apiVersion),
                VK_API_VERSION_PATCH(properties.properties.apiVersion));

    // Every core feature the device has, as the emulator enables what it finds.
    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                .pNext = &features13};
    VkPhysicalDeviceVulkan11Features features11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                                .pNext = &features12};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &features11};
    vkGetPhysicalDeviceFeatures2(gpu, &features);
    // Robust access changes code generation; the emulator leaves it as the driver offers.
    std::printf("shaderInt64 %u shaderInt16 %u shaderFloat16 %u robustBufferAccess %u\n",
                features.features.shaderInt64, features.features.shaderInt16,
                features12.shaderFloat16, features.features.robustBufferAccess);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, families.data());
    uint32_t family = 0;
    for (uint32_t index = 0; index < family_count; ++index) {
        if (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = index;
            break;
        }
    }
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkDeviceCreateInfo device_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
    };
    VkDevice device{};
    if (const VkResult result = vkCreateDevice(gpu, &device_ci, nullptr, &device);
        result != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateDevice failed: %s (%d)\n", ResultName(result), result);
        return 1;
    }

    int failures = 0;
    for (int arg = 1; arg < argc; ++arg) {
        const char* const path = argv[arg];
        const std::vector<uint32_t> words = ReadWords(path);
        const Reflection reflection = Reflect(words);
        if (!reflection.error.empty()) {
            std::printf("%s: %s\n", path, reflection.error.c_str());
            ++failures;
            continue;
        }
        std::vector<VkDescriptorSetLayout> set_layouts;
        const uint32_t set_count =
            reflection.sets.empty() ? 0 : reflection.sets.rbegin()->first + 1;
        for (uint32_t set = 0; set < set_count; ++set) {
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
            if (const auto found = reflection.sets.find(set); found != reflection.sets.end()) {
                for (const auto& [index, binding] : found->second) {
                    layout_bindings.push_back(binding);
                }
            }
            const VkDescriptorSetLayoutCreateInfo set_ci{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
                .pBindings = layout_bindings.data(),
            };
            VkDescriptorSetLayout layout{};
            vkCreateDescriptorSetLayout(device, &set_ci, nullptr, &layout);
            set_layouts.push_back(layout);
        }
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = properties.properties.limits.maxPushConstantsSize,
        };
        const VkPipelineLayoutCreateInfo layout_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = static_cast<uint32_t>(set_layouts.size()),
            .pSetLayouts = set_layouts.data(),
            .pushConstantRangeCount = reflection.push_constants ? 1U : 0U,
            .pPushConstantRanges = &push_range,
        };
        VkPipelineLayout pipeline_layout{};
        vkCreatePipelineLayout(device, &layout_ci, nullptr, &pipeline_layout);
        const VkShaderModuleCreateInfo module_ci{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = words.size() * 4,
            .pCode = words.data(),
        };
        VkShaderModule module{};
        VkResult result = vkCreateShaderModule(device, &module_ci, nullptr, &module);
        VkPipeline pipeline{};
        if (result == VK_SUCCESS) {
            const VkComputePipelineCreateInfo pipeline_ci{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = module,
                    .pName = "main",
                },
                .layout = pipeline_layout,
            };
            result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr,
                                              &pipeline);
        }
        std::printf("%s: %s (%d), %zu bindings in %u sets%s\n", path, ResultName(result), result,
                    [&] {
                        size_t total = 0;
                        for (const auto& [set, bindings] : reflection.sets) {
                            total += bindings.size();
                        }
                        return total;
                    }(),
                    set_count, reflection.push_constants ? ", push constants" : "");
        std::fflush(stdout);
        failures += result != VK_SUCCESS;
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, module, nullptr);
        }
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        for (const VkDescriptorSetLayout layout : set_layouts) {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("%d of %d failed\n", failures, argc - 1);
    return failures == 0 ? 0 : 3;
}
