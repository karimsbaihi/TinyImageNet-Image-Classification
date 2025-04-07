path = '/home/karim/Documents/3Y/ML/project/'
wnids_path=path+'data/tiny-imagenet-200/wnids.txt'
words_path=path+'data/tiny-imagenet-200/words.txt'
def parse_class_mappings(wnids_path=path+'data/tiny-imagenet-200/wnids.txt',
                         words_path=path+'data/tiny-imagenet-200/words.txt'):
    # Map WNID to class index
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    
    # Map WNID to human-readable words
    with open(words_path) as f:
        wnid_to_words = {}
        for line in f:
            wnid, desc = line.strip().split('\t')
            wnid_to_words[wnid] = desc
    return class_to_idx, wnid_to_words



# class_to_idx, wnid_to_words = parse_class_mappings()
# # Test class_to_idx mapping
# print(f"Total classes: {len(class_to_idx)}")
# print(f"Sample class_to_idx: {dict(list(class_to_idx.items())[:5])}")  # Print first 5 classes
