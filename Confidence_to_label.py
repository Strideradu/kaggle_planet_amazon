path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/vgg16_moe_4_20_epoch.csv"
output = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/submission_vgg16_moe_4_20_epoch.csv"
labels = ['blooming', 'partly_cloudy', 'water', 'habitation', 'agriculture', 'cultivation', 'conventional_mine', 'artisinal_mine', 'blow_down', 'primary', 'bare_ground', 'slash_burn', 'road', 'clear', 'selective_logging', 'cloudy', 'haze']

with open(output, "w") as fout:
    with open(path) as f:
        print >> fout, "image_name,tags"
        for line in f:
            if line[0] != "V":
                sp = line.strip().split(",")
                video_id = sp[0]
                conf_sp = sp[1].split(" ")
                result = []
                for i in range(len(conf_sp)):
                    if i % 2 != 0:
                        tag_id = int(conf_sp[i - 1])
                        score = float(conf_sp[i])
                        if score > 0.5:
                            result.append(labels[tag_id])

                print >> fout, ",".join([video_id, " ".join(result)])


