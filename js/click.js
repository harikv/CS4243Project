var data;

$(function(){
	$('#canvas').click(function(e){
		var offset = $(this).offset();
    	console.log(e.pageX - offset.left, e.pageY - offset.top);
	});
	$('#user-toolbar').toolbar({
		content: '#user-toolbar-options', 
		position: 'top'
	});
	$('#user-toolbar').on('toolbarItemClick',
	function(event, target){
		console.log('here');
		console.log(target);
		}	
	);
});

// $(function(){
// 	var canvas;
// 	var ctx;

// 	var canvasOffset;
// 	var offsetX;
// 	var offsetY;

// 	var isDrawing = false;

// 	canvas = document.getElementById('canvas');
// 	console.log(canvas);
// 	ctx = canvas.getContext("2d");

// 	canvasOffset = $("#canvas").offset();
// 	offsetX = canvasOffset.left;
// 	offsetY = canvasOffset.top;

// 	$("#canvas").on('mousedown', function (e) {
// 	    handleMouseDown(e);
// 	}).on('mouseup', function(e) {
// 	    handleMouseUp();
// 	}).on('mousemove', function(e) {
// 	    handleMouseMove(e);
// 	});


// 	var startX;
// 	var startY;

// 	function handleMouseUp() {
// 		isDrawing = false;
// 		canvas.style.cursor = "default";	
// 	}

// 	function handleMouseMove(e) {
// 		if (isDrawing) {
// 			var mouseX = parseInt(e.clientX - offsetX);
// 			var mouseY = parseInt(e.clientY - offsetY);				
			
// 			ctx.clearRect(0, 0, canvas.width, canvas.height);
// 			ctx.beginPath();
// 			ctx.rect(startX, startY, mouseX - startX, mouseY - startY);
// 			ctx.stroke();
// 		}
// 	}

// 	function handleMouseDown(e) {
// 		canvas.style.cursor = "crosshair";		
// 		isDrawing = true
// 		startX = parseInt(e.clientX - offsetX);
// 		startY = parseInt(e.clientY - offsetY);
// 	}

// });

