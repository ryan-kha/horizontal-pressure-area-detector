"""Library to test side_view_detector.py
.
Run via: 'python3 -m unittest side_view_detector_test' from console.

Test format:
def test_FILE_#_x(self):
  image_path = FILEPATH
  image_length = 
  image_height = 
  generated_results = side_view_detector.get_horizontal_pressure_areas(
    image_path=image_path,
    image_length=image_length,
    image_height = image_height,
    output_path=_OUTPUT_PATH
  )

  expected_results = side_view_detector.areas(
    a_area={
      0: 
      1: 
    },
    b_area={
      0: 
      1: 
    },
    c_area={
      0: 
      1: 
    },
    d_area={
      0: 
      1: 
    },
  )

  print("FILENAME.png")
  print("\n\nExpected results: " + str(expected_results))
  print("\nGenerated results: " + str(generated_results)) 
  
  self.assertTrue(self.verify_generated_results(
    expected_results=expected_results,
    generated_results=generated_results,
    error_percentage=.05
  ))
"""

from cgitb import small
import unittest  # Guide at: https://docs.python.org/3/library/unittest.html

import side_view_detector

import warnings
warnings.filterwarnings("ignore")

_OUTPUT_PATH = "Output"

class SideViewDetectorTest(unittest.TestCase):
  """Class to test accuracy of side view detector."""
  def verify_generated_results(self, expected_results: side_view_detector.areas,
                               generated_results: side_view_detector.areas,
                               error_percentage: float) -> bool:
    """Function to ensure side views are within error_percentage of expected."""
    if expected_results.a_area.keys() != generated_results.a_area.keys():
      print("Failed: A keys not equal")
      return False
    if expected_results.b_area.keys() != generated_results.b_area.keys():
      print("Failed: B keys not equal")
      return False
    if expected_results.c_area.keys() != generated_results.c_area.keys():
      print("Failed: C keys not equal")
      return False
    if expected_results.d_area.keys() != generated_results.d_area.keys():
      print("Failed: D keys not equal")
      return False

    for key, value in expected_results.a_area.items():
      # print("Percentage = " + str(abs(value - generated_results.a_area[key]) / value))
      if abs(value - generated_results.a_area[key]) / value > error_percentage:
        print("Failed: a_areas not equal")
        print("Expected: " + str(value))
        print("Generated: " + str(generated_results.a_area[key]))
        return False
    for key, value in expected_results.b_area.items():
      if abs(value - generated_results.b_area[key]) / value > error_percentage:
        print("Failed: b_areas not equal")
        return False
    for key, value in expected_results.c_area.items():
      if abs(value - generated_results.c_area[key]) / value > error_percentage:
        print("Failed: c_areas not equal")
        return False
    for key, value in expected_results.d_area.items():
      
      if abs(value - generated_results.d_area[key]) / value > error_percentage:
        print("Failed: d_areas not equal")
        return False

    return True

  def test_left_1_x(self):
    image_path = "Test_Images/left_1.png"
    image_length = 44.2
    image_height = 31.8
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 48,
        1: 108
      },
      b_area={
        0: 125,
        1: 0
      },
      c_area={
        0: 72,
        1: 162
      },
      d_area={
        0: 144,
        1: 0
      },
    )
    print("Left1.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))

        
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=0.05
    ))
    
  def test_left_2_x(self):
    image_path = "Test_Images/left_2.png"
    image_length = 64.32
    image_height = 31.59
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 97,
        1: 220
      },
      b_area={
        0: 121,
        1: 0
      },
      c_area={
        0: 232,
        1: 330
      },
      d_area={
        0: 97,
        1: 0
      },
    )
    
    print("Left2.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))

    """
    print("\n\nExpected results 2: " + str(expected_results))
    print("\nGenerated results 2: " + str(generated_results))
    print("\nGenerated A Keys: " + str(generated_results.a_area.keys()))
    print("Generated B Keys: " + str(generated_results.b_area.keys()))
    print("Generated C Keys: " + str(generated_results.c_area.keys()))
    print("Generated D Keys: " + str(generated_results.d_area.keys()))

    print("\nExpected A Keys: " + str(expected_results.a_area.keys()))
    print("Expected B Keys: " + str(expected_results.b_area.keys()))
    print("Expected C Keys: " + str(expected_results.c_area.keys()))
    print("Expected D Keys: " + str(expected_results.d_area.keys()))
    """
     
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_left_3_x(self):
    image_path = "Test_Images/left_3.png"
    image_length = 41.94
    image_height = 30.70
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 52,
        1: 114
      },
      b_area={
        0: 107,
        1: 0
      },
      c_area={
        0: 77.5,
        1: 160
      },
      d_area={
        0: 137,
        1: 0
      },
    )

    print("Left3.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 

    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_right_2_x(self):
    image_path = "Test_Images/right_2.png"
    image_length = 66.4635
    image_height = 32.6698
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 120.6395,
        1: 232.3359
      },
      b_area={
        0: 128.6787,
        1: 0.0
      },
      c_area={
        0: 270.1578,
        1: 330.3346
      },
      d_area={
        0: 104.1811,
        1: 0.0
      },
    )

    print("right_2.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_right_3_x(self):
    image_path = "Test_Images/right_3.png"
    image_length = 68.8797
    image_height = 42.5805
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 201.7101,
        1: 205.1531
      },
      b_area={
        0: 218.6232,
        1: 0.0
      },
      c_area={
        0: 217.1069,
        1: 149.4936
      },
      d_area={
        0: 26.2350,
        1: 0.0
      },
    )

    print("right3.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_rear_1_x(self):
    image_path = "Test_Images/rear_1.png"
    image_length = 42.65
    image_height = 33.17
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 52,
        1: 120
      },
      b_area={
        0: 94,
        1: 0
      },
      c_area={
        0: 160,
        1: 191
      },
      d_area={
        0: 32,
        1: 0
      },
    )
    print("Rear1.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))

    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_rear_2_x(self):
    image_path = "Test_Images/rear_2.png"
    image_length = 47.4815
    image_height = 37.8685
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 96.7810,
        1: 108.9605
      },
      b_area={
        0: 0.0,
        1: 0.0
      },
      c_area={
        0: 238.6981,
        1: 163.8569
      },
      d_area={
        0: 0.0,
        1: 0.0
      },
    )

    print("rear_2.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_rear_3_x(self):
    image_path = "Test_Images/rear_3.png"
    image_length = 47.6918
    image_height = 32.8774
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 159.0829,
      },
      b_area={
        0: 148.7210,
      },
      c_area={
        0: 159.5541,
      },
      d_area={
        0: 316.3836,
      },
    )

    print("rear_3.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_rear_4_x(self):

    image_path = "Test_Images/rear_4.png"
    image_length = 76.6666
    image_height = 47.4796
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 91.1353,
        1: 165.4975
      },
      b_area={
        0: 220.8649,
        1: 0.0
      },
      c_area={
        0: 85.2107,
        1: 233.4667
      },
      d_area={
        0: 303.1124,
        1: 0.0
      },
    )

    print("rear4.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_front_1_x(self):
    image_path = "Test_Images/front_1.png"
    image_length = 57.4939
    image_height = 35.6927
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height=image_height,
      output_path=_OUTPUT_PATH
    )
    
    expected_results = side_view_detector.areas(
      a_area={
        0: 67.4834,
        1: 89.7665
      },
      b_area={
        0: 83.7175,
        1: 40.2028
      },
      c_area={
        0: 117.8686,
        1: 133.5099
      },
      d_area={
        0: 86.5742,
        1: 49.7289
      },
    )
    
    print("Front1.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
  def test_front_2_x(self):
    image_path = "Test_Images/front_2.png"
    image_length = 57.9932
    image_height = 51.4660
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 97.1272,
        1: 117.8797
      },
      b_area={
        0: 33.3267,
        1: 0
      },
      c_area={
        0: 220.9980,
        1: 162.0285
      },
      d_area={
        0: 0,
        1: 0
      },
    )

    print("front_2.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results)) 
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))

  def test_front_3_x(self):
    image_path = "Test_Images/front_3.png"
    image_length = 45.57
    image_height = 32.31
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 93.2,
        1: 134.2
      },
      b_area={
        0: 0,
        1: 0
      },
      c_area={
        0: 206,
        1: 200
      },
      d_area={
        0: 0,
        1: 0
      },
    )

    print("Front3.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    )) 
    
    
  def test_mirikeen_right(self):
      image_path = "Test_Images/mirikeen_right.png"
      image_length = 98.77
      image_height = 44.0
      generated_results = side_view_detector.get_horizontal_pressure_areas(
        image_path=image_path,
        image_length=image_length,
        image_height = image_height,
        output_path=_OUTPUT_PATH
      )

      expected_results = side_view_detector.areas(
        a_area={
          0: 56,
          1: 195,
          2: 204
        },
        b_area={
          0: 50,
          1: 104,
          2: 119
        },
        c_area={
          0: 144,
          1: 316,
          2: 336
        },
        d_area={
          0: 221,
          1: 34,
          2: 66
        },
      )

      print("Mirikeen_right.png")
      print("\n\nExpected results: " + str(expected_results))
      print("\nGenerated results: " + str(generated_results))
      
      self.assertTrue(self.verify_generated_results(
        expected_results=expected_results,
        generated_results=generated_results,
        error_percentage=.05
      ))

  def test_mirikeen_front(self):
    image_path = "Test_Images/mirikeen_front.png"
    image_length = 89.18
    image_height = 40.7
    generated_results = side_view_detector.get_horizontal_pressure_areas(
      image_path=image_path,
      image_length=image_length,
      image_height = image_height,
      output_path=_OUTPUT_PATH
    )

    expected_results = side_view_detector.areas(
      a_area={
        0: 9,
        1: 89,
        2: 193
      },
      b_area={
        0: 24,
        1: 55,
        2: 99
      },
      c_area={
        0: 181,
        1: 312,
        2: 398
      },
      d_area={
        0: 49,
        1: 57,
        2: 8
      },
    )

    print("Mirikeen_front.png")
    print("\n\nExpected results: " + str(expected_results))
    print("\nGenerated results: " + str(generated_results))
    
    self.assertTrue(self.verify_generated_results(
      expected_results=expected_results,
      generated_results=generated_results,
      error_percentage=.05
    ))
    
    
def main():
    test = SideViewDetectorTest()
    test.test_left_1_x()
    test.test_left_2_x()
    test.test_left_3_x()
    test.test_front_1_x()
    test.test_front_2_x()
    test.test_front_3_x()
    test.test_rear_1_x()
    test.test_rear_2_x()
    test.test_rear_3_x()
    test.test_rear_4_x()
    test.test_right_2_x()
    test.test_right_3_x()
    test.test_mirikeen_front()
    test.test_mirikeen_right()

if __name__ == "__main__":
    main()