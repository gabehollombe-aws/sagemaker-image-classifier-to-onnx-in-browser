// argMax via https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
export function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

export function zipArrays(a, b) {
  return a.map((e, i) => [e, b[i]])
}

// scaleImage via https://gist.github.com/MikeRogers0/6264546
export function scaleImage(url, width, height, callback){
	let img = new window.Image();

	img.onload = function(){
		var canvas = document.createElement("canvas"),
        ctx = canvas.getContext("2d");

        canvas.width = width;
        canvas.height = height;

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // resize code via https://sdqali.in/blog/2013/10/03/fitting-an-image-in-to-a-canvas-object/
        var imageAspectRatio = img.width / img.height;
        var canvasAspectRatio = canvas.width / canvas.height;
        var renderableHeight, renderableWidth, xStart, yStart;
        if(imageAspectRatio < canvasAspectRatio) {
          renderableHeight = canvas.height;
          renderableWidth = img.width * (renderableHeight / img.height);
          xStart = (canvas.width - renderableWidth) / 2;
          yStart = 0;
        }
        else if(imageAspectRatio > canvasAspectRatio) {
          renderableWidth = canvas.width
          renderableHeight = img.height * (renderableWidth / img.width);
          xStart = 0;
          yStart = (canvas.height - renderableHeight) / 2;
        }
        else {
          renderableHeight = canvas.height;
          renderableWidth = canvas.width;
          xStart = 0;
          yStart = 0;
        }
        ctx.drawImage(img, xStart, yStart, renderableWidth, renderableHeight);

        callback(canvas);
	};

  img.src = url;
}