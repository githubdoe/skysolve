//<!-- startup status timeout if auto status is checked at start of pages-->



sks = {}
sks.consts = {

    yes: 0,
    no: 1,

    shutter: function() {
        var x =  $('#shutterSelect');
        return x;

    },
    ISO: function() {
        var x = $('#ISOSelect');
        return x;
    },
    frame: function() {
        var x = $('#frameSelect');
        return x;
    },
    format: function() {
        var x = $('#formatSelect');
        return x;
    },
    saveObs: function() {
        var x = $('#saveObs');
        return x;
    },

    demoMode: false,
    showHistory: 0,
    showSolution: 0
};

function setIniShutter( shutVal){
    console.log("shutter value", shutVal)
    sks.consts.shutter().val(shutVal);
}
function setIniISO( ISOVal){
    console.log("ISO val", ISOVal)
    sks.consts.ISO().val(ISOVal);
}
function setIniFrame( frameVal){
    sks.consts.frame().val(frameVal);
}
function setIniFormat(formatVal){
    sks.consts.format().val(formatVal);
}

$(document).ready(function(){

    var checkBox = document.getElementById("autoStatusCB");
    if (checkBox.checked == true) {
        setTimeout(updateStatusField, 1000);
    }

    var cb = document.getElementById('showStars')
    var btn = document.getElementById('ShowStars')
    if (cb.checked == false){
        btn.disabled = true
    }
    // event hookups/subscribes
    sks.consts.shutter().change(
        function() {
            let x = sks.consts.shutter().val();
            $.post("/setShutter/" + x, data = {
                suggest: x
            }, function(result) {});
    });

    sks.consts.ISO().change(
        function() {

            let x = sks.consts.ISO().val();
            console.log("ISO is being set", x);
            $.post("/setISO/" + x, data = {
                suggest: x
            }, function(result) {});
    });

    sks.consts.frame().change(
        function() {
            console.log("frame size change");
            let x = sks.consts.frame().val();
            $.post("/setFrame/" + x, data = {
                suggest: x
            }, function(result) {});
    });

    sks.consts.format().change(
        function() {
            let x = sks.consts.format().val();
            $.post("/setFormat/" + x, data = {
                suggest: x
            }, function(result) {});
    });

    sks.consts.saveObs().change(
        function() {
            let x = sks.consts.saveObs().val();
            $.post("/saveObs/" + x, data = {
                suggest:x
            }, function(result) {});            
    });

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

    $('#idPause').click (
        function() {
            ajax_get_Status('/pause')
        })
    $('#idAlign').click (
        function() {
            ajax_get_Status('/Align')
        })
    $('#idSolve').click (
        function() {
            ajax_get_Status('/Solve')
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
    function showReplaybuttons(show){
        var x = document.getElementById("stepNext");
        var y = document.getElementById("stepPrev");
        var z = document.getElementById("solveThis");
        var zz = document.getElementById("retryNext");
        if (show) {
            x.style.display = "inline";
            y.style.display = "inline";
            z.style.display = "inline";
            zz.style.display = "inline";
        }  else {
            x.style.display = "none";
            y.style.display = "none";
            z.style.display = "none";
            zz.style.display = "none"; 
        }              
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

    $('#clearObsLog').click(
        function() {
            ajax_get_Status('/clearObsLog')
        })
    

    $('#solveThis').click(
        function() {
            console.log("imageStep pressed")
            ajax_get_Status('/solveThis')
        })
    $('#clearImages').click(
        function(){
            ajax_get_Status('/clearImages')
        }
    )
    
    $('#demoMode').click(
        function(){
            if (sks.consts.demoMode) {
                showReplaybuttons(false);
                sks.consts.demoMode = false;
            }
            else {
                ajax_get_Status('/demoMode');
                showReplaybuttons(true);
                sks.consts.demoMode=true;
            }
        }
    )
    
    $('#testMode').click(
        function() {
            
            if (!sks.consts.demoMode)
                ajax_get_Status('/testMode');
            var x = document.getElementById("stepNext");
            if (x.style.display === "none") {
                showReplaybuttons(true);
            }                   
            else {
                showReplaybuttons(false);
            }
        })
    $('#showStars').click(

            function() {
                var cb = document.getElementById('showStars')
                var btn = document.getElementById('ShowStars')

                if (cb.checked == false) {
                    btn.disabled = true

                }
                else {
                    btn.disabled = false

                }
            }
        )
    $('#ClearLog').click(
        function(){
            console.log("check changed")
            var btn = document.getElementById('ShowStars')
            btn.disabled = true
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
    $('#saveImages').click(
        function() {

            var checkBox = document.getElementById("saveImages");
            if (checkBox.checked == true) {
                sks.consts.showSolution = 1;
            }
            else {
                sks.consts.showSolution = 0;
            }
            let x = sks.consts.showSolution;

            $.post("/saveImages/" + x, data = {
                suggest: x
            }, function(result) {});
        }
    ) 


});



