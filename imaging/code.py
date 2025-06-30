
slm_size_x = 1600
slm_size_y = 1152
macro_pixel_size = 32
# macro_pixel_size = 16
no_macropixel_x = int(round(slm_size_x / macro_pixel_size))
no_macropixel_y = int(round(slm_size_y / macro_pixel_size))
resvoir_nodes = int(no_macropixel_x * no_macropixel_y)
print(resvoir_nodes)
# slm_cord_x_range = np.arange(0, 1921, macro_pixel_size)
# slm_cord_y_range = np.arange(0, 1201, macro_pixel_size)
slm_cord_x_range = np.arange(0, (slm_size_x+1), macro_pixel_size)
slm_cord_y_range = np.arange(0, (slm_size_y+1), macro_pixel_size)
ini_manager = IniFileManager(r"C:\Users\User\PycharmProjects\DecisionMakingSystem\Tools\resources\parameter_settings.ini")
ini_manager.read_ini_file()
x1 = int(ini_manager.get_value("CCD_Image_Map", "start_x"))
x2 = int(ini_manager.get_value("CCD_Image_Map", "end_x"))
y1 = int(ini_manager.get_value("CCD_Image_Map", "start_y"))
y2 = int(ini_manager.get_value("CCD_Image_Map", "end_y"))

ccd_cord_x_range, ccd_cord_y_range, cropped_x_range, cropped_y_range = generate_ccd_coord(x1, x2, y1, y2,
                                                                                          no_macropixel_x,
                                                                                          no_macropixel_y)
average_ccd_output = block_average_1(image, no_macropixel_x, no_macropixel_y, cropped_x_range, cropped_y_range)


def block_average_1(CCD_Img, no_macro_x, no_macro_y, cropped_X, cropped_Y):

    avg_array = []

    for i in range(0, no_macro_x):
        for j in range(0, no_macro_y):

                x1 = int((cropped_X[i])[0])
                x2 = int((cropped_X[i])[1])
                y1 = int((cropped_Y[j])[0])
                y2 = int((cropped_Y[j])[1])
                avg_value = np.mean(CCD_Img[y1:y2, x1:x2])
                avg_array.append(avg_value)

    return np.array(avg_array)

def generate_ccd_coord(x_start, x_end, y_start, y_end, no_macro_x, no_macro_y):


    CCD_Macro_No_x = no_macro_x
    CCD_Macro_No_y = no_macro_y

    ccd_cord_x_range = np.round(np.linspace(x_start, x_end, CCD_Macro_No_x))
    ccd_cord_y_range = np.round(np.linspace(y_start, y_end, CCD_Macro_No_y))


    each_macro_size_x = np.array([[x+10, x+54] for x in ccd_cord_x_range])
    each_macro_size_y = np.array([[y+10, y+54] for y in ccd_cord_y_range])

    each_macro_size_x = np.array(each_macro_size_x)
    each_macro_size_y = np.array(each_macro_size_y)

    return ccd_cord_x_range, ccd_cord_y_range, each_macro_size_x, each_macro_size_y

