materials_txt = "D:\\Study\\code_proj\\ThRend-master\\flymaterials.txt"

def writer_to_txt(emissivity1, diffuse1):
    with open(materials_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip() == "name steel":
            emissivity_parts = lines[i + 2].split()
            emissivity_parts[-1] = str(emissivity1)
            lines[i + 2] = " ".join(emissivity_parts) + "\n"
            diffuse_parts = lines[i + 3].split()
            diffuse_parts[-1] = str(diffuse1)
            lines[i + 3] = " ".join(diffuse_parts) + "\n"
            # roughness_parts = lines[i + 4].split()
            # roughness_parts[-1] = str(x3)
            # lines[i + 4] = " ".join(roughness_parts) + "\n"

    with open(materials_txt, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    print("Updated mal")