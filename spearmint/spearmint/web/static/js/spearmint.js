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

    var yAxis = d3.svg.axis()
	.scale(y)
	.orient("left");

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
        .attr("x1", x(-0.2))
        .attr("y2", function(d) { return -1 * y(d); })
        .attr("x2", x(0));

    svg.append("g")
	.attr("class", "y axis")
	.call(yAxis)
	.append("text")
	.attr("transform", "rotate(-90)")
	.attr("y", 2)
	.attr("dy", ".71em")
	.style("text-anchor", "end")
	.text("Temperature (ºF)");

    /*
       svg.append("text")
       .attr("x", width / 2)
       .attr("y",  height / 2)
       .style("text-anchor", "middle")
       .text("experiments");
       */
}

function bar_chart(div_id, data, maxval) {
    var x = d3.scale.linear()
       .domain([0, maxval])
       .range(["0px", "420px"]);

    var y = d3.scale.ordinal()
       .domain(data)
       .rangeBands([0, 30*data.length]);
    var chart = d3.select(div_id).append("svg")
      .attr("class", "chart")
      .attr("width", 440)
      .attr("height", 33*data.length)
      .style("margin-left", "32px") // Tweak alignment…
      .append("g")
      .attr("transform", "translate(10,15)");

  chart.selectAll("line")
      .data(x.ticks(10))
      .enter().append("line")
      .attr("x1", x)
      .attr("x2", x)
      .attr("y1", 0)
      .attr("y2", 30*data.length)
      .style("stroke", "#ccc");

  chart.selectAll(".rule")
      .data(x.ticks(5))
      .enter().append("text")
      .attr("class", "rule")
      .attr("x", x)
      .attr("y", 0)
      .attr("dy", -3)
      .attr("text-anchor", "middle")
      .text(String);

  chart.selectAll("rect")
      .data(data)
      .enter().append("rect")
      .attr("y", y)
      .attr("width", x)
      .attr("height", y.rangeBand());

  chart.selectAll(".bar")
      .data(data)
      .enter().append("text")
      .attr("class", "bar")
      .attr("x", x)
      .attr("y", function(d) { return y(d) + y.rangeBand() / 2; })
      .attr("dx", -3)
      .attr("dy", ".35em")
      .attr("text-anchor", "end")
      .text(String);

  chart.append("line")
      .attr("y1", 0)
      .attr("y2", 30*data.length)
      .style("stroke", "#000");
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
