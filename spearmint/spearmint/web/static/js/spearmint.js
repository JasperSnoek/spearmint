function line_chart(height, width, margin, y_data) {
    var min_y = d3.min([0, d3.min(y_data)]);
    var max_y = d3.min([max_score, d3.max(y_data)]);

    var min_x = 0;
    var max_x = y_data.length;

    var x = d3.scale.linear()
        .domain([min_x, max_x])
        .range([0 + margin, width - margin]);

    var y = d3.scale.linear()
        .domain([min_y, max_y])
        .range([0 + margin, height - margin]);

    $("#result-chart").empty();
    var vis = d3.select("#result-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)

        var g = vis.append("svg:g")
        .attr("transform", "translate(0, " + height + ")");

    var line = d3.svg.line()
        .x(function(d,i) { return x(i); })
        .y(function(d) { return -1 * y(d); });

    g.append("svg:path").attr("d", line(y_data));

    g.append("svg:line")
        .attr("x1", x(0))
        .attr("y1", -1 * y(0))
        .attr("x2", x(width))
        .attr("y2", -1 * y(0));

    g.append("svg:line")
        .attr("x1", x(0))
        .attr("y1", -1 * y(0))
        .attr("x2", x(0))
        .attr("y2", -1 * y(d3.max(y_data)));

    g.selectAll(".xLabel")
        .data(x.ticks(5))
        .enter().append("svg:text")
        .attr("class", "xLabel")
        .text(String)
        .attr("x", function(d) { return x(d) })
        .attr("y", 0)
        .attr("text-anchor", "middle");

    g.selectAll(".yLabel")
        .data(y.ticks(4))
        .enter().append("svg:text")
        .attr("class", "yLabel")
        .text(String)
        .attr("x", 0)
        .attr("y", function(d) { return -1 * y(d) })
        .attr("text-anchor", "right")
        .attr("dy", 4);

    g.selectAll(".xTicks")
        .data(x.ticks(5))
        .enter().append("svg:line")
        .attr("class", "xTicks")
        .attr("x1", function(d) { return x(d); })
        .attr("y1", -1 * y(0))
        .attr("x2", function(d) { return x(d); })
        .attr("y2", -1 * y(-0.3));

    g.selectAll(".yTicks")
        .data(y.ticks(4))
        .enter().append("svg:line")
        .attr("class", "yTicks")
        .attr("y1", function(d) { return -1 * y(d); })
        .attr("x1", x(-0.3))
        .attr("y2", function(d) { return -1 * y(d); })
        .attr("x2", x(0));

    /*
       svg.append("text")
       .attr("x", width / 2)
       .attr("y",  height / 2)
       .style("text-anchor", "middle")
       .text("experiments");
       */
}

var REFRESH_RATE = 2000;
var width = 400;
var height = 200;
var margin = 20;

function load_status() {
    $('#status').load('/status');
}

var refresher = setInterval(load_status, REFRESH_RATE);
load_status();
var max_score = 10000000000000;

$('#max-score').change(function() {
    max_score = $('#max-score').val();
});
