import cv2
import numpy as np
import imutils
import base64
import cv2
import io

from imutils.perspective import four_point_transform
from imutils import contours


def count_circles_per_line(questionCnts, y_tolerance=10):
        # Initialize a list to store the center y-coordinates of contours
        centers_y = []

        # Loop over the question contours
        for c in questionCnts:
            # Compute the bounding box of the contour
            (_, y, _, h) = cv2.boundingRect(c)
            
            # Compute the center y-coordinate of the contour
            center_y = y + (h // 2)
            
            # Append the center y-coordinate
            centers_y.append(center_y)

        # Sort the center y-coordinates
        centers_y.sort()

        # Initialize variables to hold the count of circles per line
        numberOfOptions = []
        current_line_y = centers_y[0]
        current_count = 1

        # Loop through sorted center y-coordinates
        for i in range(1, len(centers_y)):
            # If the next center is within the y_tolerance, it's on the same line
            if abs(centers_y[i] - current_line_y) <= y_tolerance:
                current_count += 1
            else:
                # If it's not on the same line, store the count and reset it
                numberOfOptions.append(current_count)
                current_count = 1
                current_line_y = centers_y[i]

        # Append the last line's count
        numberOfOptions.append(current_count)

        return numberOfOptions

def filter_contours_by_average(questionCnts,avgNumberOfOptions, y_tolerance=10):
    # Count the circles per line using the existing function
    
    # Calculate the average number of circles per line
    
    # Group contours by their y-coordinates
    lines = {}
    for c in questionCnts:
        # Compute the bounding box and center y-coordinate
        (_, y, _, h) = cv2.boundingRect(c)
        center_y = y + (h // 2)
        
        # Find which line the center belongs to
        found_line = False
        for line_y in lines.keys():
            if abs(center_y - line_y) <= y_tolerance:
                lines[line_y].append(c)
                found_line = True
                break
        
        # If not found, initialize a new line
        if not found_line:
            lines[center_y] = [c]


    # Filter out lines that don't match the average number of options
    filtered_lines = {line_y: cnts for line_y, cnts in lines.items() if len(cnts) == avgNumberOfOptions}
    
    # Flatten the list of contours
    filtered_contours = [c for cnts in filtered_lines.values() for c in cnts]
    
    return filtered_contours


def process_bubble_sheet(image,ANSWER_KEY, already_in_frame=False ):

    print("starting processing")
    print(ANSWER_KEY)
    print(already_in_frame)
    # Read the image file into bytes
    image_bytes = image.read()

    # Convert the bytes to a NumPy array
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)

    # - Convert to grayscale
    gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('gray_image.png', gray)

    # - Blur the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # - Find edges
    edged = cv2.Canny(blurred, 75, 200)

    cv2.imwrite('edged_image.png', edged)

    if not already_in_frame:

        # - Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        docCnt = None
        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4:
                    docCnt = approx
                    #print(docCnt)
                    x, y, w, h = cv2.boundingRect(docCnt)
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                    break

        # - Perform perspective transformation
        try:
            warped = four_point_transform(gray, docCnt.reshape(4, 2))
        except:
            print("An error occured in the transform, perhaps try marking it as a full image ")
    
    else:
        warped = gray

    cv2.imwrite('warped_image.png', warped)

    # - Apply thresholding
    thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    # - Find and evaluate the bubbles

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20:# and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    test = warped.copy()  #  use copy() to avoid drawing on the original image

    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

    for contour in questionCnts:
        x, y, w, h = cv2.boundingRect(contour)

        # Compute the center of the contour
        center = (int(x + w/2), int(y + h/2))

        # Compute radius as half of the average of width and height
        radius = int((w + h) / 4)

        # Draw the circle around the contour


        cv2.circle(test, center, radius, (0, 0, 255), 2)

    cv2.imwrite('test_image.png', test)

    #Count averages and filter away useless contours    
    counts_per_line = count_circles_per_line(questionCnts)
    numberOfLines = len(counts_per_line)

    # Calculate the average number of options per line
    if counts_per_line:
        numberOfOptions= round(sum(counts_per_line) / numberOfLines)
    else:
        numberOfOptions = 0  # Or handle this case as an error if it's unexpected

    #print("Number of circles per line:", numberOfOptions)

    # Filter contours based on the average number of options
    questionCnts = filter_contours_by_average(questionCnts, numberOfOptions)
    # Sort the contours

    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    visualized = warped.copy()

    for (q, i) in enumerate(np.arange(0, len(questionCnts), numberOfOptions)):
        cnts = contours.sort_contours(questionCnts[i:i + numberOfOptions])[0]
        bubbled = None
        
        for (j, c) in enumerate(cnts):
            # Draw the contour on the image
            cv2.drawContours(visualized, [c], -1, (0, 255, 0), 2)
            
            # Label the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(visualized, "{}-{}".format(q + 1, j + 1), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            

    cv2.imwrite('visualized_image.png', visualized)


    # Return the processed results
    # Count the correct answers
    correct = 0

    # Ensure `warped` is a color image.
    if len(warped.shape) == 2 or warped.shape[2] == 1:  # this checks if the image is grayscale
        visualised_all = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    else:
        visualised_all = warped.copy()

    # First pass: Calculate the average non-zero pixel count for each option
    totals = []

    for (q, i) in enumerate(np.arange(0, len(questionCnts), numberOfOptions)):
        cnts = contours.sort_contours(questionCnts[i:i + numberOfOptions])[0]    
        #print(f"Question {q + 1}  has {len(cnts)} options detected.")

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            #print(print(f"Question {q + 1} {j+1} has {total} nonZero"))
            totals.append(total)

    # Calculate the average non-zero pixel count across all bubbles
    average_non_zero = np.mean(totals)

    #print(average_non_zero)

    # Second pass: Determine the filled bubbles and grade

    # Update the second pass to handle skipped questions and ensure correct number of bubbles
    for (q, i) in enumerate(np.arange(0, len(questionCnts), numberOfOptions)):
        # Check if the current segment has the correct number of options
        if i + numberOfOptions <= len(questionCnts):
            cnts = contours.sort_contours(questionCnts[i:i + numberOfOptions])[0]
            bubbled = None

            # Loop over the sorted contours for the current question
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)

                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                # If total number of non-zero pixels is greater than average, consider it filled
                if total > average_non_zero/1:
                    if bubbled is None or total > bubbled[0]:
                        bubbled = (total, j)

            # Determine the correct answer for the current question
            correct_answer = ANSWER_KEY.get(q, None)

            # Proceed only if there is a correct answer for the question
            if correct_answer is not None:
                # Draw the correct/incorrect answer on the visualised image
                if bubbled is not None:
                    color = (0, 0, 255)  # Incorrect answer is red
                    if bubbled[1] == correct_answer:
                        color = (0, 255, 0)  # Correct answer is green
                        correct += 1
                    cv2.drawContours(visualised_all, [cnts[bubbled[1]]], -1, color, 3)
                    if bubbled[1] != correct_answer and len(cnts) > correct_answer:
                        cv2.drawContours(visualised_all, [cnts[correct_answer]], -1, (0, 255, 255), 3)  # Corrected answer is yellow
                else:
                    print("Skipped: not bubbled")
                    # Handle the case where nothing is filled in for a line (skipped question)
                    # You can draw a different color or add a specific mark to indicate a skipped question.
                    # For example, we'll use blue color to indicate a skipped question.
                    cv2.putText(visualised_all, "SKIPPED", (cnts[0][0][0][0], cnts[0][0][0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                print("Skipped: correct_answer is None")
                # Handle the case where nothing is filled in for a line (skipped question)
                # You can draw a different color or add a specific mark to indicate a skipped question.
                # For example, we'll use blue color to indicate a skipped question.
                cv2.putText(visualised_all, "SKIPPED", (cnts[0][0][0][0], cnts[0][0][0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
        else:
            # Handle the case where the number of detected bubbles is not equal to numberOfOptions
            print(f"Warning: Question {q+1} does not have the correct number of detected options.")


    # Print the total number of correct answers
    #print(f"Total correct answers: {correct}")

    _, buffer = cv2.imencode('.png', visualised_all)
    img_str = base64.b64encode(buffer).decode()


    #results dict
    results = {
        'score': correct,
        'numberOfQuestions': numberOfLines,
        'visualised': img_str,
    }

    cv2.imwrite("final.png", visualised_all)

    return results


def main():
    # Define the path to your test image
    test_image_path = './Test_images/9.png'

    # Load your test image
    test_image = cv2.imread(test_image_path)

    # Check if the image was loaded correctly
    if test_image is None:
        print(f"Error: The image at {test_image_path} could not be loaded.")
        return
    
    # Convert the image to bytes, as expected by process_bubble_sheet
    # This step is simulating reading the image as bytes, as would be the case in a Flask app
    _, buffer = cv2.imencode('.jpg', test_image)
    image_bytes = buffer.tobytes()

    # Call your process_bubble_sheet function with the image bytes
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
    results = process_bubble_sheet(image_bytes, ANSWER_KEY)

    filename = "visualised_results.png"

    cv2.imwrite(filename, results['visualised'])

    # Print the results
    print(f"{results['score']}/{results['numberOfQuestions']}" )

    # Optionally, if you want to see the result image, uncomment the following lines:
    # cv2.imshow("Processed Image", results['visualised_all'])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()