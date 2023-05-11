using OpenCvSharp;

var hight = 1920;
var width = 1080;
var ran = new Random(DateTime.Now.Millisecond);

var vw = new VideoWriter(@"C:\Users\k_tak\OneDrive\デスクトップ\sunaarashi.mp4",
    FourCC.H264, 10, new Size(width, hight));

var mat = new Mat(hight, width, MatType.CV_8UC3);

for (var k = 0; k < 40; k++)
{
    for (var i = 0; i < mat.Height; i = i + 5)
    {
        for (var j = 0; j < mat.Width; j = j + 5)
        {
            var b = BitConverter.GetBytes(ran.Next(255))[0];
            var g = BitConverter.GetBytes(ran.Next(255))[0];
            var r = BitConverter.GetBytes(ran.Next(255))[0];

            for (var ii = i; ii < i + 5; ii++)
            {
                for (var jj = j; jj < j + 5; jj++)
                {
                    mat.Set(ii, jj, new Vec3b(b, g, r));
                }
            }
        }
    }
    vw.Write(mat);
}

vw.Release();
vw.Dispose();
mat.Dispose();