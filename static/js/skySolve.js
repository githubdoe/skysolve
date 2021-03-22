//<!-- startup status timeout if auto status is checked at start of pages-->



sks = {}
sks.consts = {

    yes: 0,
    no: 1,

    shutter: function() {
        var x =  $('#shutterSelect');
        return x;

    },
    showHistory: 0,
    showSolution: 0
};

$(document).ready(function(){

    var checkBox = document.getElementById("autoStatusCB");
    console.log("the check box is ", checkBox)
    if (checkBox.checked == true) {
        console.log("checked is true ", checkBox)
        setTimeout(updateStatusField, 5000);
    }


    // event hookups/subscribes
    sks.consts.shutter().change(
        function() {
            let x = sks.consts.shutter().val();
            $.post("/setShutter/" + x, data = {
                suggest: x
            }, function(result) {});
    });
    // starting values
    sks.consts.shutter().val(1);

    $('#stepNext').click(
        function() {
            ajax_get_Status('/nextImage')
        })

    $('#stepPrev').click(
        function() {
            ajax_get_Status('/prevImage')
        })

    $('#solveThis').click(
        function() {
            console.log("imageStep pressed")
            ajax_get_Status('/solveThis')
        })

    $('#testMode').click(
        function() {
            ajax_get_Status('/testMode');
            var x = document.getElementById("stepNext");
            var y = document.getElementById("stepPrev");
            var z = document.getElementById("solveThis");
            if (x.style.display === "none") {
                x.style.display = "inline";
                y.style.display = "inline";
                z.style.display = "inline";
            }                   
            else {
                x.style.display = "none";
                y.style.display = "none";
                z.style.display = "none";
            }
        })
        $('#ClearHistory').click(
            function(){
                var text = document.getElementById('solveStatusText');
                text.innerHTML = "";
            }
        )
        $('#showSolutionCB').click(
            function() {
                console.log( sks.consts.showSolution)
                var checkBox = document.getElementById("showSolutionCB");
                if (checkBox.checked == true) {
                    sks.consts.showSolution = 1;
                }
                else {
                    sks.consts.showSolution = 0;
                }
                let x = sks.consts.showSolution;
                console.log(x)
                $.post("/showSolution/" + x, data = {
                    suggest: x
                }, function(result) {});
            }
        ) 


});



