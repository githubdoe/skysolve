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

    function ajax_get_Status(cmdroute) {
        $.ajax({
            url: cmdroute,
            method: 'POST',
            success: function(result) {
                document.getElementById("statusField").innerHTML = result;
            }
        });
    }

    $('#stepNext').click(
        function() {
            ajax_get_Status('/nextImage')
        })

    $('#retryNext').click(
        function() {
            ajax_get_Status('/retryImage')
        })

    function ajax_get_Obs(cmdroute) {
        $.ajax({
            url: cmdroute,
            method: 'POST',
            success: function(result) {
                console.log("result",result);
                var txt = document.getElementById("currentObs");
                txt.value = result
            }
        });
    }

    $('#startObs').click(
        function() {
            console.log("start obs");
            ajax_get_Obs('/startObs')
        })

    $('#nextObs').click(
        function() {
            ajax_get_Obs('/nextObs')
        })

    $('#prevObs').click(
        function() {
            ajax_get_Obs('/prevObs')
        })

    $('#stepPrev').click(
        function() {
            ajax_get_Obs('/prevImage')
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
            var zz = document.getElementById("retryNext");
            if (x.style.display === "none") {
                x.style.display = "inline";
                y.style.display = "inline";
                z.style.display = "inline";
                zz.style.display = "inline";
            }                   
            else {
                x.style.display = "none";
                y.style.display = "none";
                z.style.display = "none";
                zz.style.display = "none";
            }
        })
        $('#ClearLog').click(
            function(){
                var text = document.getElementById('solveStatusText');
                text.innerHTML = "";
            }
        )
        $('#showSolutionCB').click(
            function() {

                var checkBox = document.getElementById("showSolutionCB");
                if (checkBox.checked == true) {
                    sks.consts.showSolution = 1;
                }
                else {
                    sks.consts.showSolution = 0;
                }
                let x = sks.consts.showSolution;

                $.post("/showSolution/" + x, data = {
                    suggest: x
                }, function(result) {});
            }
        ) 


});



