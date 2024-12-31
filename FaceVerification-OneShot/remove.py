import pickle

# Đọc dữ liệu từ file pickle
with open(r'data.pickle', 'rb') as handle:
    database = pickle.load(handle)

# Lấy danh sách các khóa trước khi xóa
keys_to_remove = list(database.keys())

# Xóa và in từng phần tử
if len(keys_to_remove) == 0:
    print("Nobody")
else:
    for key in keys_to_remove :
        print(f"Removed {key} from the database.")
        del database[key]

# Lưu lại dữ liệu đã thay đổi vào file pickle
with open(r'data.pickle', 'wb') as handle:
    pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

