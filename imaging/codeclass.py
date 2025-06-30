import numpy as np
import cv2
import configparser  # needed for IniFileManager

class IniFileManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = configparser.ConfigParser()

    def read_ini_file(self):
        self.config.read(self.filepath)

    def get_value(self, section, key):
        return self.config.get(section, key)

class CCDProcessor:
    def __init__(self, image):
        self.slm_size_x = 1600
        self.slm_size_y = 1152
        self.macro_pixel_size = 32
        # self.macro_pixel_size = 16

        self.no_macropixel_x = int(round(self.slm_size_x / self.macro_pixel_size))
        self.no_macropixel_y = int(round(self.slm_size_y / self.macro_pixel_size))
        self.resvoir_nodes = int(self.no_macropixel_x * self.no_macropixel_y)
        print(self.resvoir_nodes)

        # self.slm_cord_x_range = np.arange(0, 1921, macro_pixel_size)
        # self.slm_cord_y_range = np.arange(0, 1201, macro_pixel_size)
        self.slm_cord_x_range = np.arange(0, (self.slm_size_x + 1), self.macro_pixel_size)
        self.slm_cord_y_range = np.arange(0, (self.slm_size_y + 1), self.macro_pixel_size)

        self.ini_manager = IniFileManager(
            r"/Users/muskaan_garg_/BioTrackAI/parameter_settings.ini"  
        )
        self.ini_manager.read_ini_file()

        self.x1 = int(self.ini_manager.get_value("CCD_Image_Map", "start_x"))
        self.x2 = int(self.ini_manager.get_value("CCD_Image_Map", "end_x"))
        self.y1 = int(self.ini_manager.get_value("CCD_Image_Map", "start_y"))
        self.y2 = int(self.ini_manager.get_value("CCD_Image_Map", "end_y"))

        self.ccd_cord_x_range, self.ccd_cord_y_range, self.cropped_x_range, self.cropped_y_range = \
            self.generate_ccd_coord(self.x1, self.x2, self.y1, self.y2,
                                    self.no_macropixel_x, self.no_macropixel_y)

        self.image = image
        self.average_ccd_output = self.block_average_1(self.image, self.no_macropixel_x,
                                                       self.no_macropixel_y,
                                                       self.cropped_x_range,
                                                       self.cropped_y_range)

    def block_average_1(self, CCD_Img, no_macro_x, no_macro_y, cropped_X, cropped_Y):
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

    def generate_ccd_coord(self, x_start, x_end, y_start, y_end, no_macro_x, no_macro_y):
        CCD_Macro_No_x = no_macro_x
        CCD_Macro_No_y = no_macro_y

        ccd_cord_x_range = np.round(np.linspace(x_start, x_end, CCD_Macro_No_x))
        ccd_cord_y_range = np.round(np.linspace(y_start, y_end, CCD_Macro_No_y))

        each_macro_size_x = np.array([[x + 10, x + 54] for x in ccd_cord_x_range])
        each_macro_size_y = np.array([[y + 10, y + 54] for y in ccd_cord_y_range])

        each_macro_size_x = np.array(each_macro_size_x)
        each_macro_size_y = np.array(each_macro_size_y)

        return ccd_cord_x_range, ccd_cord_y_range, each_macro_size_x, each_macro_size_y


image = cv2.imread("download.png", cv2.IMREAD_GRAYSCALE)
ccd = CCDProcessor(image)

# Print all coordinate outputs
print("ccd_cord_x_range:\n", ccd.ccd_cord_x_range)
print("ccd_cord_y_range:\n", ccd.ccd_cord_y_range)
print("cropped_x_range:\n", ccd.cropped_x_range)
print("cropped_y_range:\n", ccd.cropped_y_range)

# Print block average output
print("\nAverage CCD output values:", ccd.average_ccd_output)

# Shape (for verification)
print("\nOutput shape:", ccd.average_ccd_output.shape)
