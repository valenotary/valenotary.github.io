---
layout: page
title: Posts
permalink: /posts/
---
<!-- <ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul> -->

<ul>
    {% assign sorted = site.posts | sort: 'date' | reverse %}
    {% for post in sorted %}
    <li>
    <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
</ul>
