using OpenCvSharp;

const string inputVideo = @"C:\Users\k_tak\Downloads\『Googleフォト』スタビライズ　手ブレ補正前 歩きver. - from YouTube.mp4";
const string outputVideo = @"C:\Users\k_tak\Downloads\StabilizeVideo.mp4";

StabilizeVideo(inputVideo, outputVideo);

void StabilizeVideo(string inputVideo, string outputVideo)
{
    using var capture = new VideoCapture(inputVideo);
    if (!capture.IsOpened())
    {
        Console.WriteLine("動画ファイルを開けませんでした。");
        return;
    }

    var width = (int)capture.Get(VideoCaptureProperties.FrameWidth);
    var height = (int)capture.Get(VideoCaptureProperties.FrameHeight);
    var fps = capture.Get(VideoCaptureProperties.Fps);

    var writer = new VideoWriter(outputVideo, FourCC.H264, fps, new Size(width, height));
    if (!writer.IsOpened())
    {
        Console.WriteLine("出力ファイルを開けませんでした。");
        return;
    }

    var prevGray = new Mat();
    var prev = new Mat();

    Mat H = Mat.Eye(3, 3, MatType.CV_32F);

    capture.Read(prev);
    Cv2.CvtColor(prev, prevGray, ColorConversionCodes.BGR2GRAY);

    var frame = new Mat();

    while (true)
    {
        capture.Read(frame); // read next frame from the video file
        if (frame.Empty()) // if the frame is empty, it means the video is over
            break;

        // 1. GoodFeaturesToTrack: Detect feature points
        using var frameGray = new Mat();
        Cv2.CvtColor(frame, frameGray, ColorConversionCodes.BGR2GRAY);
        var prevPts = Cv2.GoodFeaturesToTrack(frameGray, 200, 0.03, 7, null, 7, false, 0.04);

        // 2. CalcOpticalFlowPyrLK: Calculate optical flow
        var nextImg = frameGray.Clone();
        var nextPts = new Point2f[prevPts.Length];
        Cv2.CalcOpticalFlowPyrLK(frameGray, nextImg, prevPts, ref nextPts, out var status, out _);

        // 3. FindHomography: Compute perspective transformation
        var inliers = new List<Point2d>();
        var outliers = new List<Point2d>();
        for (var i = 0; i < status.Length; i++)
            if (status[i] == 1)
            {
                inliers.Add(new Point2d(prevPts[i].X, prevPts[i].Y));
                outliers.Add(new Point2d(nextPts[i].X, nextPts[i].Y));
            }

        // ensure we have enough points to calculate the homography
        if (inliers.Count >= 4 && outliers.Count >= 4)
        {
            var homography = Cv2.FindHomography(inliers, outliers, HomographyMethods.Ransac);

            // 4. WarpPerspective: Apply perspective transformation to the image
            var result = new Mat();
            Cv2.WarpPerspective(frame, result, homography, frame.Size());

            writer.Write(result); // write the output frame to the video file
        }
        else
        {
            // If not enough points, skip this frame
            Console.WriteLine("Not enough points for homography. Skipping this frame.");
        }
    }

    writer.Release();
    writer.Dispose();
    frame.Dispose();

}