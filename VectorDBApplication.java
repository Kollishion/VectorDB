
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

public class VectorDBApplication {

	static final int DIMS = 16;

	static class VectorItem {
		int id;
		String metadata, category;
		float[] emb;

		VectorItem(int id, String metadata, String category, float[] emb) {
			this.id = id;
			this.metadata = metadata;
			this.category = category;
			this.emb = emb;
		}
	}

	interface DistFn {
		float distance(float[] a, float[] b);
	}

	static float euclidean(float[] a, float[] b) {
		float s = 0;
		for (int i = 0; i < a.length; i++) {
			float d = a[i] - b[i];
			s += d * d;
		}
		return (float) Math.sqrt(s);
	}

	static float cosine(float[] a, float[] b) {
		float dot = 0, na = 0, nb = 0;
		for (int i = 0; i < a.length; i++) {
			dot += a[i] * b[i];
			na += a[i] * a[i];
			nb += b[i] * b[i];
		}
		if (na < 1e-9f || nb < 1e-9f)
			return 1.0f;
		return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
	}

	static float manhattan(float[] a, float[] b) {
		float s = 0;
		for (int i = 0; i < a.length; i++)
			s += Math.abs(a[i] - b[i]);
		return s;
	}

	static DistFn getDistFn(String m) {
		return switch (m) {
			case "cosine" -> VectorDBApplication::cosine;
			case "manhattan" -> VectorDBApplication::manhattan;
			default -> VectorDBApplication::euclidean;
		};
	}

	static class Pair {
		float distance;
		int id;

		Pair(float distance, int id) {
			this.distance = distance;
			this.id = id;
		}
	}

	static class BruteForce {
		List<VectorItem> items = new ArrayList<>();

		void insert(VectorItem v) {
			items.add(v);
		}

		List<Pair> knn(float[] q, int k, DistFn dist) {
			List<Pair> r = new ArrayList<>();
			for (VectorItem v : items)
				r.add(new Pair(dist.distance(q, v.emb), v.id));
			r.sort(Comparator.comparingDouble(x -> x.distance));
			if (r.size() > k)
				r = new ArrayList<>(r.subList(0, k));
			return r;
		}

		void remove(int id) {
			items.removeIf(v -> v.id == id);
		}
	}

	static class KDNode {
		VectorItem item;
		KDNode left, right;

		KDNode(VectorItem v) {
			item = v;
		}
	}

	static class KDTree {
		KDNode root;
		int dims;

		KDTree(int dims) {
			this.dims = dims;
		}

		KDNode insertNode(KDNode n, VectorItem v, int d) {
			if (n == null)
				return new KDNode(v);
			int ax = d % dims;
			if (v.emb[ax] < n.item.emb[ax])
				n.left = insertNode(n.left, v, d + 1);
			else
				n.right = insertNode(n.right, v, d + 1);
			return n;
		}

		void insert(VectorItem v) {
			root = insertNode(root, v, 0);
		}

		void knn(
				KDNode n,
				float[] q,
				int k,
				int d,
				DistFn dist,
				PriorityQueue<Pair> heap) {
			if (n == null)
				return;

			float dn = dist.distance(q, n.item.emb);

			if (heap.size() < k || dn < heap.peek().distance) {
				heap.offer(new Pair(dn, n.item.id));
				if (heap.size() > k)
					heap.poll();
			}

			int ax = d % dims;
			float diff = q[ax] - n.item.emb[ax];

			KDNode closer = diff < 0 ? n.left : n.right;
			KDNode farther = diff < 0 ? n.right : n.left;

			knn(closer, q, k, d + 1, dist, heap);

			if (heap.size() < k || Math.abs(diff) < heap.peek().distance)
				knn(farther, q, k, d + 1, dist, heap);
		}

		List<Pair> knn(float[] q, int k, DistFn dist) {
			if (k <= 0 || root == null)
				return new ArrayList<>();

			PriorityQueue<Pair> heap = new PriorityQueue<>(Comparator.comparingDouble(x -> -x.distance));

			knn(root, q, k, 0, dist, heap);

			List<Pair> r = new ArrayList<>();
			while (!heap.isEmpty())
				r.add(heap.poll());

			r.sort(Comparator.comparingDouble(x -> x.distance));
			return r;
		}

		void rebuild(Collection<VectorItem> items) {
			root = null;
			for (VectorItem v : items)
				insert(v);
		}
	}

	static class HNSW {
		static class Node {
			VectorItem item;
			int maxLyr;
			List<List<Integer>> nbrs;

			Node(VectorItem item, int maxLyr) {
				this.item = item;
				this.maxLyr = maxLyr;
				this.nbrs = new ArrayList<>();
				for (int i = 0; i <= maxLyr; i++)
					this.nbrs.add(new ArrayList<>());
			}
		}

		Map<Integer, Node> graph = new HashMap<>();
		int M, M0, efBuild;
		float mL;
		int topLayer = -1;
		int entryPt = -1;
		Random rng = new Random(42);

		HNSW() {
			this(16, 200);
		}

		HNSW(int m, int efBuild) {
			M = m;
			M0 = 2 * m;
			this.efBuild = efBuild;
			mL = 1.0f / (float) Math.log(m);
		}

		int randLevel() {
			double u = rng.nextDouble();
			if (u == 0)
				u = Double.MIN_VALUE;
			return (int) Math.floor(-Math.log(u) * mL);
		}

		List<Pair> searchLayer(float[] q, int ep, int ef, int layer, DistFn dist) {
			Set<Integer> visited = new HashSet<>();

			PriorityQueue<Pair> candidates = new PriorityQueue<>(
					Comparator.comparingDouble(x -> x.distance));

			PriorityQueue<Pair> found = new PriorityQueue<>(Comparator.comparingDouble(x -> -x.distance));

			Node start = graph.get(ep);
			if (start == null)
				return new ArrayList<>();

			float d0 = dist.distance(q, start.item.emb);

			visited.add(ep);
			candidates.offer(new Pair(d0, ep));
			found.offer(new Pair(d0, ep));

			while (!candidates.isEmpty()) {
				Pair current = candidates.poll();

				if (found.size() >= ef && current.distance > found.peek().distance)
					break;

				Node currentNode = graph.get(current.id);
				if (currentNode == null || layer >= currentNode.nbrs.size())
					continue;

				for (int nid : currentNode.nbrs.get(layer)) {
					if (visited.contains(nid))
						continue;

					Node neighbor = graph.get(nid);
					if (neighbor == null)
						continue;

					visited.add(nid);

					float nd = dist.distance(q, neighbor.item.emb);

					if (found.size() < ef || nd < found.peek().distance) {
						candidates.offer(new Pair(nd, nid));
						found.offer(new Pair(nd, nid));

						if (found.size() > ef)
							found.poll();
					}
				}
			}

			List<Pair> result = new ArrayList<>();
			while (!found.isEmpty())
				result.add(found.poll());

			result.sort(Comparator.comparingDouble(x -> x.distance));
			return result;
		}

		List<Integer> selectNbrs(List<Pair> candidates, int maxM) {
			List<Integer> result = new ArrayList<>();
			for (int i = 0; i < Math.min(candidates.size(), maxM); i++)
				result.add(candidates.get(i).id);
			return result;
		}

		void insert(VectorItem item, DistFn dist) {
			int id = item.id;
			int lvl = randLevel();

			graph.put(id, new Node(item, lvl));

			if (entryPt == -1) {
				entryPt = id;
				topLayer = lvl;
				return;
			}

			int ep = entryPt;

			for (int layer = topLayer; layer > lvl; layer--) {
				Node epNode = graph.get(ep);

				if (epNode != null && layer < epNode.nbrs.size()) {
					List<Pair> w = searchLayer(item.emb, ep, 1, layer, dist);
					if (!w.isEmpty())
						ep = w.get(0).id;
				}
			}

			for (int layer = Math.min(topLayer, lvl); layer >= 0; layer--) {
				List<Pair> w = searchLayer(item.emb, ep, efBuild, layer, dist);

				int maxM = layer == 0 ? M0 : M;
				List<Integer> selected = selectNbrs(w, maxM);

				graph.get(id).nbrs.set(layer, selected);

				for (int nid : selected) {
					Node neighbor = graph.get(nid);
					if (neighbor == null)
						continue;

					while (neighbor.nbrs.size() <= layer)
						neighbor.nbrs.add(new ArrayList<>());

					List<Integer> conn = neighbor.nbrs.get(layer);
					conn.add(id);

					if (conn.size() > maxM) {
						List<Pair> distances = new ArrayList<>();

						for (int c : conn) {
							Node candidate = graph.get(c);
							if (candidate != null) {
								distances.add(new Pair(
										dist.distance(neighbor.item.emb,
												candidate.item.emb),
										c));
							}
						}

						distances.sort(Comparator.comparingDouble(x -> x.distance));
						conn.clear();

						for (int i = 0; i < Math.min(maxM, distances.size()); i++)
							conn.add(distances.get(i).id);
					}
				}

				if (!w.isEmpty())
					ep = w.get(0).id;
			}

			if (lvl > topLayer) {
				topLayer = lvl;
				entryPt = id;
			}
		}

		List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
			if (entryPt == -1 || k <= 0)
				return new ArrayList<>();

			int ep = entryPt;

			for (int layer = topLayer; layer > 0; layer--) {
				Node epNode = graph.get(ep);

				if (epNode != null && layer < epNode.nbrs.size()) {
					List<Pair> w = searchLayer(q, ep, 1, layer, dist);
					if (!w.isEmpty())
						ep = w.get(0).id;
				}
			}

			List<Pair> result = searchLayer(q, ep, Math.max(ef, k), 0, dist);

			if (result.size() > k)
				result = new ArrayList<>(result.subList(0, k));

			return result;
		}

		void remove(int id) {
			if (!graph.containsKey(id))
				return;

			for (Node node : graph.values()) {
				for (List<Integer> layer : node.nbrs)
					layer.removeIf(x -> x == id);
			}

			if (entryPt == id) {
				entryPt = -1;

				for (int nid : graph.keySet()) {
					if (nid != id) {
						entryPt = nid;
						break;
					}
				}

				if (entryPt != -1)
					topLayer = graph.get(entryPt).maxLyr;
				else
					topLayer = -1;
			}

			graph.remove(id);
		}

		static class GraphInfo {
			int topLayer;
			int nodeCount;
			List<Integer> nodesPerLayer = new ArrayList<>();
			List<Integer> edgesPerLayer = new ArrayList<>();
			List<NodeView> nodes = new ArrayList<>();
			List<EdgeView> edges = new ArrayList<>();
		}

		static class NodeView {
			int id;
			String metadata;
			String category;
			int maxLyr;

			NodeView(int id, String metadata, String category, int maxLyr) {
				this.id = id;
				this.metadata = metadata;
				this.category = category;
				this.maxLyr = maxLyr;
			}
		}

		static class EdgeView {
			int src, dst, lyr;

			EdgeView(int src, int dst, int lyr) {
				this.src = src;
				this.dst = dst;
				this.lyr = lyr;
			}
		}

		GraphInfo getInfo() {
			GraphInfo info = new GraphInfo();

			info.topLayer = topLayer;
			info.nodeCount = graph.size();

			int maxL = Math.max(topLayer + 1, 1);

			for (int i = 0; i < maxL; i++) {
				info.nodesPerLayer.add(0);
				info.edgesPerLayer.add(0);
			}

			for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
				int id = entry.getKey();
				Node node = entry.getValue();

				info.nodes.add(new NodeView(
						id,
						node.item.metadata,
						node.item.category,
						node.maxLyr));

				for (int layer = 0; layer <= node.maxLyr && layer < maxL; layer++) {

					info.nodesPerLayer.set(
							layer,
							info.nodesPerLayer.get(layer) + 1);

					if (layer < node.nbrs.size()) {
						for (int nid : node.nbrs.get(layer)) {
							if (id < nid) {
								info.edgesPerLayer.set(
										layer,
										info.edgesPerLayer.get(layer) + 1);

								info.edges.add(
										new EdgeView(id, nid, layer));
							}
						}
					}
				}
			}

			return info;
		}

		int size() {
			return graph.size();
		}
	}

	static class VectorDB {
		Map<Integer, VectorItem> store = new HashMap<>();
		BruteForce bf = new BruteForce();
		KDTree kdt;
		HNSW hnsw = new HNSW(16, 200);
		ReentrantLock lock = new ReentrantLock();
		int nextId = 1;
		final int dims;

		VectorDB(int dims) {
			this.dims = dims;
			kdt = new KDTree(dims);
		}

		int insert(
				String metadata,
				String category,
				float[] emb,
				DistFn dist) {
			lock.lock();

			try {
				VectorItem v = new VectorItem(nextId++, metadata, category, emb);

				store.put(v.id, v);

				bf.insert(v);
				kdt.insert(v);
				hnsw.insert(v, dist);

				return v.id;
			} finally {
				lock.unlock();
			}
		}

		boolean remove(int id) {
			lock.lock();

			try {
				if (!store.containsKey(id))
					return false;

				store.remove(id);
				bf.remove(id);
				hnsw.remove(id);

				kdt.rebuild(store.values());

				return true;
			} finally {
				lock.unlock();
			}
		}

		static class Hit {
			int id;
			String meta;
			String cat;
			float[] emb;
			float dist;

			Hit(
					int id,
					String meta,
					String cat,
					float[] emb,
					float dist) {
				this.id = id;
				this.meta = meta;
				this.cat = cat;
				this.emb = emb;
				this.dist = dist;
			}
		}

		static class SearchOut {
			List<Hit> hits = new ArrayList<>();
			long us;
			String algo;
			String metric;
		}

		SearchOut search(
				float[] q,
				int k,
				String metric,
				String algo) {
			lock.lock();

			try {
				DistFn dfn = getDistFn(metric);

				long t0 = System.nanoTime();

				List<Pair> raw;

				if (algo.equals("bruteforce"))
					raw = bf.knn(q, k, dfn);
				else if (algo.equals("kdtree"))
					raw = kdt.knn(q, k, dfn);
				else
					raw = hnsw.knn(q, k, 50, dfn);

				long us = (System.nanoTime() - t0) / 1000;

				SearchOut out = new SearchOut();

				out.us = us;
				out.algo = algo;
				out.metric = metric;

				for (Pair p : raw) {
					VectorItem v = store.get(p.id);

					if (v != null) {
						out.hits.add(
								new Hit(
										v.id,
										v.metadata,
										v.category,
										v.emb,
										p.distance));
					}
				}

				return out;
			} finally {
				lock.unlock();
			}
		}

		static class BenchOut {
			long bfUs, kdUs, hnswUs;
			int n;

			BenchOut(long bfUs, long kdUs, long hnswUs, int n) {
				this.bfUs = bfUs;
				this.kdUs = kdUs;
				this.hnswUs = hnswUs;
				this.n = n;
			}
		}

		BenchOut benchmark(
				float[] q,
				int k,
				String metric) {
			lock.lock();

			try {
				DistFn dfn = getDistFn(metric);

				long bfStart = System.nanoTime();
				bf.knn(q, k, dfn);
				long bfUs = (System.nanoTime() - bfStart) / 1000;

				long kdStart = System.nanoTime();
				kdt.knn(q, k, dfn);
				long kdUs = (System.nanoTime() - kdStart) / 1000;

				long hnswStart = System.nanoTime();
				hnsw.knn(q, k, 50, dfn);
				long hnswUs = (System.nanoTime() - hnswStart) / 1000;

				return new BenchOut(
						bfUs,
						kdUs,
						hnswUs,
						store.size());
			} finally {
				lock.unlock();
			}
		}

		List<VectorItem> all() {
			lock.lock();

			try {
				return new ArrayList<>(store.values());
			} finally {
				lock.unlock();
			}
		}

		HNSW.GraphInfo hnswInfo() {
			lock.lock();

			try {
				return hnsw.getInfo();
			} finally {
				lock.unlock();
			}
		}

		int size() {
			lock.lock();

			try {
				return store.size();
			} finally {
				lock.unlock();
			}
		}
	}

	static String json(String s) {
		if (s == null)
			return "\"\"";

		StringBuilder out = new StringBuilder("\"");

		for (char c : s.toCharArray()) {
			switch (c) {
				case '"' -> out.append("\\\"");
				case '\\' -> out.append("\\\\");
				case '\n' -> out.append("\\n");
				case '\r' -> out.append("\\r");
				case '\t' -> out.append("\\t");
				default -> out.append(c);
			}
		}

		return out.append('"').toString();
	}

	static String json(float[] v) {
		StringBuilder sb = new StringBuilder("[");

		for (int i = 0; i < v.length; i++) {
			if (i > 0)
				sb.append(',');
			sb.append(String.format(Locale.US, "%.4f", v[i]));
		}

		return sb.append(']').toString();
	}

	static float[] parseVec(String s) {
		List<Float> values = new ArrayList<>();

		for (String token : s.split(",")) {
			try {
				values.add(Float.parseFloat(token.trim()));
			} catch (Exception ignored) {
			}
		}

		float[] result = new float[values.size()];

		for (int i = 0; i < values.size(); i++)
			result[i] = values.get(i);

		return result;
	}

	static String body(HttpExchange exchange) throws IOException {
		return new String(
				exchange.getRequestBody().readAllBytes(),
				StandardCharsets.UTF_8);
	}

	static void response(
			HttpExchange exchange,
			int status,
			String content) throws IOException {

		exchange.getResponseHeaders().set(
				"Access-Control-Allow-Origin",
				"*");

		exchange.getResponseHeaders().set(
				"Access-Control-Allow-Methods",
				"GET, POST, DELETE, OPTIONS");

		exchange.getResponseHeaders().set(
				"Access-Control-Allow-Headers",
				"Content-Type");

		exchange.getResponseHeaders().set(
				"Content-Type",
				"application/json");

		byte[] bytes = content.getBytes(StandardCharsets.UTF_8);

		exchange.sendResponseHeaders(status, bytes.length);

		try (OutputStream os = exchange.getResponseBody()) {
			os.write(bytes);
		}
	}

	static String extractString(String body, String key) {
		String needle = "\"" + key + "\"";

		int p = body.indexOf(needle);

		if (p == -1)
			return "";

		p = body.indexOf(':', p);

		if (p == -1)
			return "";

		p++;

		while (p < body.length() &&
				Character.isWhitespace(body.charAt(p)))
			p++;

		if (p >= body.length() || body.charAt(p) != '"')
			return "";

		p++;

		StringBuilder result = new StringBuilder();

		while (p < body.length()) {
			char c = body.charAt(p);

			if (c == '"')
				break;

			if (c == '\\' && p + 1 < body.length()) {
				p++;

				switch (body.charAt(p)) {
					case '"' -> result.append('"');
					case '\\' -> result.append('\\');
					case 'n' -> result.append('\n');
					case 'r' -> result.append('\r');
					case 't' -> result.append('\t');
					default -> result.append(body.charAt(p));
				}
			} else {
				result.append(c);
			}

			p++;
		}

		return result.toString();
	}

	static int extractInt(
			String body,
			String key,
			int def) {
		String needle = "\"" + key + "\"";

		int p = body.indexOf(needle);

		if (p == -1)
			return def;

		p = body.indexOf(':', p);

		if (p == -1)
			return def;

		p++;

		while (p < body.length() &&
				Character.isWhitespace(body.charAt(p)))
			p++;

		int end = p;

		while (end < body.length() &&
				(Character.isDigit(body.charAt(end)) ||
						body.charAt(end) == '-'))
			end++;

		try {
			return Integer.parseInt(body.substring(p, end));
		} catch (Exception e) {
			return def;
		}
	}

	static float[] extractArray(String body, String key) {
		String needle = "\"" + key + "\"";

		int p = body.indexOf(needle);

		if (p == -1)
			return new float[0];

		p = body.indexOf('[', p);

		if (p == -1)
			return new float[0];

		int e = body.indexOf(']', p);

		if (e == -1)
			return new float[0];

		return parseVec(body.substring(p + 1, e));
	}

	static boolean parseBody(
			String body,
			StringHolder metadata,
			StringHolder category,
			FloatArrayHolder embedding) {
		metadata.value = extractString(body, "metadata");
		category.value = extractString(body, "category");
		embedding.value = extractArray(body, "embedding");

		return !metadata.value.isEmpty() &&
				embedding.value.length > 0;
	}

	static class StringHolder {
		String value;
	}

	static class FloatArrayHolder {
		float[] value;
	}

	static List<String> chunkText(
			String text,
			int chunkWords,
			int overlapWords) {
		String[] words = text.trim().split("\\s+");

		if (words.length == 0 ||
				(words.length == 1 && words[0].isEmpty()))
			return new ArrayList<>();

		if (words.length <= chunkWords)
			return List.of(text);

		List<String> chunks = new ArrayList<>();

		int step = chunkWords - overlapWords;

		for (int i = 0; i < words.length; i += step) {
			int end = Math.min(
					i + chunkWords,
					words.length);

			String chunk = String.join(
					" ",
					Arrays.copyOfRange(words, i, end));

			chunks.add(chunk);

			if (end == words.length)
				break;
		}

		return chunks;
	}

	static class OllamaClient {
		String host;
		int port;

		String embedModel = "nomic-embed-text";
		String genModel = "llama3.2";

		HttpClient client = HttpClient.newBuilder()
				.connectTimeout(Duration.ofSeconds(3))
				.build();

		OllamaClient() {
			this("127.0.0.1", 11434);
		}

		OllamaClient(String host, int port) {
			this.host = host;
			this.port = port;
		}

		String esc(String s) {
			return json(s);
		}

		boolean isAvailable() {
			try {
				HttpRequest request = HttpRequest.newBuilder()
						.uri(URI.create(
								"http://" +
										host +
										":" +
										port +
										"/api/tags"))
						.timeout(Duration.ofSeconds(2))
						.GET()
						.build();

				HttpResponse<String> response = client.send(
						request,
						HttpResponse.BodyHandlers.ofString());

				return response.statusCode() == 200;
			} catch (Exception e) {
				return false;
			}
		}

		float[] parseEmbedding(String body) {
			int p = body.indexOf("\"embedding\"");

			if (p == -1)
				return new float[0];

			p = body.indexOf('[', p);

			if (p == -1)
				return new float[0];

			int e = body.indexOf(']', p);

			if (e == -1)
				return new float[0];

			return parseVec(
					body.substring(p + 1, e));
		}

		String embed(String text) {
			try {
				String requestBody = "{\"model\":" +
						json(embedModel) +
						",\"prompt\":" +
						json(text) +
						"}";

				HttpRequest request = HttpRequest.newBuilder()
						.uri(URI.create(
								"http://" +
										host +
										":" +
										port +
										"/api/embeddings"))
						.timeout(Duration.ofSeconds(30))
						.header(
								"Content-Type",
								"application/json")
						.POST(
								HttpRequest.BodyPublishers.ofString(
										requestBody))
						.build();

				HttpResponse<String> response = client.send(
						request,
						HttpResponse.BodyHandlers.ofString());

				if (response.statusCode() != 200)
					return "";

				float[] embedding = parseEmbedding(response.body());

				return Arrays.toString(embedding);
			} catch (Exception e) {
				return "";
			}
		}

		float[] embedVector(String text) {
			try {
				String requestBody = "{\"model\":" +
						json(embedModel) +
						",\"prompt\":" +
						json(text) +
						"}";

				HttpRequest request = HttpRequest.newBuilder()
						.uri(URI.create(
								"http://" +
										host +
										":" +
										port +
										"/api/embeddings"))
						.timeout(Duration.ofSeconds(30))
						.header(
								"Content-Type",
								"application/json")
						.POST(
								HttpRequest.BodyPublishers.ofString(
										requestBody))
						.build();

				HttpResponse<String> response = client.send(
						request,
						HttpResponse.BodyHandlers.ofString());

				if (response.statusCode() != 200)
					return new float[0];

				return parseEmbedding(response.body());
			} catch (Exception e) {
				return new float[0];
			}
		}

		String generate(String prompt) {
			try {
				String requestBody = "{\"model\":" +
						json(genModel) +
						",\"prompt\":" +
						json(prompt) +
						",\"stream\":false}";

				HttpRequest request = HttpRequest.newBuilder()
						.uri(URI.create(
								"http://" +
										host +
										":" +
										port +
										"/api/generate"))
						.timeout(Duration.ofSeconds(180))
						.header(
								"Content-Type",
								"application/json")
						.POST(
								HttpRequest.BodyPublishers.ofString(
										requestBody))
						.build();

				HttpResponse<String> response = client.send(
						request,
						HttpResponse.BodyHandlers.ofString());

				if (response.statusCode() != 200)
					return "ERROR: Ollama unavailable. Run: ollama serve";

				return extractString(
						response.body(),
						"response");
			} catch (Exception e) {
				return "ERROR: Ollama unavailable. Run: ollama serve";
			}
		}
	}

	static class DocItem {
		int id;
		String title;
		String text;
		float[] emb;

		DocItem(
				int id,
				String title,
				String text,
				float[] emb) {
			this.id = id;
			this.title = title;
			this.text = text;
			this.emb = emb;
		}
	}

	static class DocumentDB {
		Map<Integer, DocItem> store = new HashMap<>();
		HNSW hnsw = new HNSW(16, 200);
		BruteForce bf = new BruteForce();
		ReentrantLock lock = new ReentrantLock();

		int nextId = 1;
		int dims = 0;

		int insert(
				String title,
				String text,
				float[] emb) {
			lock.lock();

			try {
				if (dims == 0)
					dims = emb.length;

				DocItem item = new DocItem(
						nextId++,
						title,
						text,
						emb);

				store.put(item.id, item);

				VectorItem vi = new VectorItem(
						item.id,
						title,
						"doc",
						emb);

				hnsw.insert(vi, VectorDBApplication::cosine);
				bf.insert(vi);

				return item.id;
			} finally {
				lock.unlock();
			}
		}

		static class DocHit {
			float distance;
			DocItem item;

			DocHit(float distance, DocItem item) {
				this.distance = distance;
				this.item = item;
			}
		}

		List<DocHit> search(
				float[] q,
				int k,
				float maxDist) {
			lock.lock();

			try {
				if (store.isEmpty())
					return new ArrayList<>();

				List<Pair> raw = store.size() < 10
						? bf.knn(
								q,
								k,
								VectorDBApplication::cosine)
						: hnsw.knn(
								q,
								k,
								50,
								VectorDBApplication::cosine);

				List<DocHit> result = new ArrayList<>();

				for (Pair p : raw) {
					DocItem item = store.get(p.id);

					if (item != null &&
							p.distance <= maxDist)
						result.add(
								new DocHit(
										p.distance,
										item));
				}

				return result;
			} finally {
				lock.unlock();
			}
		}

		boolean remove(int id) {
			lock.lock();

			try {
				if (!store.containsKey(id))
					return false;

				store.remove(id);
				hnsw.remove(id);
				bf.remove(id);

				return true;
			} finally {
				lock.unlock();
			}
		}

		List<DocItem> all() {
			lock.lock();

			try {
				return new ArrayList<>(store.values());
			} finally {
				lock.unlock();
			}
		}

		int size() {
			lock.lock();

			try {
				return store.size();
			} finally {
				lock.unlock();
			}
		}

		int getDims() {
			return dims;
		}
	}

	static void loadDemo(VectorDB db) {
		DistFn dist = VectorDBApplication::cosine;

		db.insert(
				"Linked List: nodes connected by pointers",
				"cs",
				new float[] {
						.90f, .85f, .72f, .68f,
						.12f, .08f, .15f, .10f,
						.05f, .08f, .06f, .09f,
						.07f, .11f, .08f, .06f
				},
				dist);

		db.insert(
				"Binary Search Tree: O(log n) search and insert",
				"cs",
				new float[] {
						.88f, .82f, .78f, .74f,
						.15f, .10f, .08f, .12f,
						.06f, .07f, .08f, .05f,
						.09f, .06f, .07f, .10f
				},
				dist);

		db.insert(
				"Dynamic Programming: memoization overlapping subproblems",
				"cs",
				new float[] {
						.82f, .76f, .88f, .80f,
						.20f, .18f, .12f, .09f,
						.07f, .06f, .08f, .07f,
						.08f, .09f, .06f, .07f
				},
				dist);

		db.insert(
				"Graph BFS and DFS: breadth and depth first traversal",
				"cs",
				new float[] {
						.85f, .80f, .75f, .82f,
						.18f, .14f, .10f, .08f,
						.06f, .09f, .07f, .06f,
						.10f, .08f, .09f, .07f
				},
				dist);

		db.insert(
				"Hash Table: O(1) lookup with collision chaining",
				"cs",
				new float[] {
						.87f, .78f, .70f, .76f,
						.13f, .11f, .09f, .14f,
						.08f, .07f, .06f, .08f,
						.07f, .10f, .08f, .09f
				},
				dist);

		db.insert(
				"Calculus: derivatives integrals and limits",
				"math",
				new float[] {
						.12f, .15f, .18f, .10f,
						.91f, .86f, .78f, .72f,
						.08f, .06f, .07f, .09f,
						.07f, .08f, .06f, .10f
				},
				dist);

		db.insert(
				"Linear Algebra: matrices eigenvalues eigenvectors",
				"math",
				new float[] {
						.20f, .18f, .15f, .12f,
						.88f, .90f, .82f, .76f,
						.09f, .07f, .08f, .06f,
						.10f, .07f, .08f, .09f
				},
				dist);

		db.insert(
				"Probability: distributions random variables Bayes theorem",
				"math",
				new float[] {
						.15f, .12f, .20f, .18f,
						.84f, .80f, .88f, .82f,
						.07f, .08f, .06f, .10f,
						.09f, .06f, .09f, .08f
				},
				dist);

		db.insert(
				"Number Theory: primes modular arithmetic RSA cryptography",
				"math",
				new float[] {
						.22f, .16f, .14f, .20f,
						.80f, .85f, .76f, .90f,
						.08f, .09f, .07f, .06f,
						.08f, .10f, .07f, .06f
				},
				dist);

		db.insert(
				"Combinatorics: permutations combinations generating functions",
				"math",
				new float[] {
						.18f, .20f, .16f, .14f,
						.86f, .78f, .84f, .80f,
						.06f, .07f, .09f, .08f,
						.06f, .09f, .10f, .07f
				},
				dist);

		db.insert(
				"Neapolitan Pizza: wood-fired dough San Marzano tomatoes",
				"food",
				new float[] {
						.08f, .06f, .09f, .07f,
						.07f, .08f, .06f, .09f,
						.90f, .86f, .78f, .72f,
						.08f, .06f, .09f, .07f
				},
				dist);

		db.insert(
				"Sushi: vinegared rice raw fish and nori rolls",
				"food",
				new float[] {
						.06f, .08f, .07f, .09f,
						.09f, .06f, .08f, .07f,
						.86f, .90f, .82f, .76f,
						.07f, .09f, .06f, .08f
				},
				dist);

		db.insert(
				"Ramen: noodle soup with chashu pork and soft-boiled eggs",
				"food",
				new float[] {
						.09f, .07f, .06f, .08f,
						.08f, .09f, .07f, .06f,
						.82f, .78f, .90f, .84f,
						.09f, .07f, .08f, .06f
				},
				dist);

		db.insert(
				"Tacos: corn tortillas with carnitas salsa and cilantro",
				"food",
				new float[] {
						.07f, .09f, .08f, .06f,
						.06f, .07f, .09f, .08f,
						.78f, .82f, .86f, .90f,
						.06f, .08f, .07f, .09f
				},
				dist);

		db.insert(
				"Croissant: laminated pastry with buttery flaky layers",
				"food",
				new float[] {
						.06f, .07f, .10f, .09f,
						.10f, .06f, .07f, .10f,
						.85f, .80f, .76f, .82f,
						.09f, .07f, .10f, .06f
				},
				dist);

		db.insert(
				"Basketball: fast-paced shooting dribbling slam dunks",
				"sports",
				new float[] {
						.09f, .07f, .08f, .10f,
						.08f, .09f, .07f, .06f,
						.08f, .07f, .09f, .06f,
						.91f, .85f, .78f, .72f
				},
				dist);

		db.insert(
				"Football: tackles touchdowns field goals and strategy",
				"sports",
				new float[] {
						.07f, .09f, .06f, .08f,
						.09f, .07f, .10f, .08f,
						.07f, .09f, .08f, .07f,
						.87f, .89f, .82f, .76f
				},
				dist);

		db.insert(
				"Tennis: racket volleys groundstrokes and Wimbledon serves",
				"sports",
				new float[] {
						.08f, .06f, .09f, .07f,
						.07f, .08f, .06f, .09f,
						.09f, .06f, .07f, .08f,
						.83f, .80f, .88f, .82f
				},
				dist);

		db.insert(
				"Chess: openings endgames tactics strategic board game",
				"sports",
				new float[] {
						.25f, .20f, .22f, .18f,
						.22f, .18f, .20f, .15f,
						.06f, .08f, .07f, .09f,
						.80f, .84f, .78f, .90f
				},
				dist);

		db.insert(
				"Swimming: butterfly freestyle backstroke Olympic competition",
				"sports",
				new float[] {
						.06f, .08f, .07f, .09f,
						.08f, .06f, .09f, .07f,
						.10f, .08f, .06f, .07f,
						.85f, .82f, .86f, .80f
				},
				dist);
	}

	static String vectorResultsJson(VectorDB.SearchOut out) {
		StringBuilder ss = new StringBuilder();

		ss.append("{\"results\":[");

		for (int i = 0; i < out.hits.size(); i++) {
			if (i > 0)
				ss.append(',');

			VectorDB.Hit h = out.hits.get(i);

			ss.append("{")
					.append("\"id\":").append(h.id)
					.append(",\"metadata\":").append(json(h.meta))
					.append(",\"category\":").append(json(h.cat))
					.append(",\"distance\":")
					.append(String.format(Locale.US, "%.6f", h.dist))
					.append(",\"embedding\":").append(json(h.emb))
					.append("}");
		}

		ss.append("]")
				.append(",\"latencyUs\":").append(out.us)
				.append(",\"algo\":").append(json(out.algo))
				.append(",\"metric\":").append(json(out.metric))
				.append("}");

		return ss.toString();
	}

	static class ServerApp {
		VectorDB db = new VectorDB(DIMS);
		DocumentDB docDB = new DocumentDB();
		OllamaClient ollama = new OllamaClient();

		void start() throws Exception {
			loadDemo(db);

			boolean ollamaUp = ollama.isAvailable();

			System.out.println("=== VectorDB Engine ===");
			System.out.println("http://localhost:8081");
			System.out.println(
					db.size() +
							" demo vectors | " +
							DIMS +
							" dims | HNSW+KD-Tree+BruteForce");
			System.out.println(
					"Ollama: " +
							(ollamaUp ? "ONLINE" : "OFFLINE"));

			HttpServer server = HttpServer.create(
					new InetSocketAddress("0.0.0.0", 8081),
					0);

			server.createContext("/", this::handle);

			server.setExecutor(
					Executors.newCachedThreadPool());

			server.start();
		}

		void handle(HttpExchange ex) throws IOException {
			String method = ex.getRequestMethod();
			String path = ex.getRequestURI().getPath();

			if (method.equals("OPTIONS")) {
				response(ex, 204, "");
				return;
			}

			try {
				if (method.equals("GET") &&
						path.equals("/")) {
					File file = new File("index.html");

					if (!file.exists()) {
						response(ex, 404, "");
						return;
					}

					String html = new String(
							java.nio.file.Files.readAllBytes(
									file.toPath()),
							StandardCharsets.UTF_8);

					ex.getResponseHeaders().set(
							"Content-Type",
							"text/html");

					byte[] bytes = html.getBytes(StandardCharsets.UTF_8);

					ex.sendResponseHeaders(
							200,
							bytes.length);

					try (OutputStream os = ex.getResponseBody()) {
						os.write(bytes);
					}

					return;
				}

				if (method.equals("GET") &&
						path.equals("/stats")) {
					response(
							ex,
							200,
							"{\"count\":" +
									db.size() +
									",\"dims\":" +
									DIMS +
									",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],"
									+
									"\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
					return;
				}

				if (method.equals("GET") &&
						path.equals("/status")) {

					boolean up = ollama.isAvailable();

					response(
							ex,
							200,
							"{\"ollamaAvailable\":" +
									up +
									",\"embedModel\":" +
									json(ollama.embedModel) +
									",\"genModel\":" +
									json(ollama.genModel) +
									",\"docCount\":" +
									docDB.size() +
									",\"docDims\":" +
									docDB.getDims() +
									",\"demoDims\":" +
									DIMS +
									",\"demoCount\":" +
									db.size() +
									"}");

					return;
				}

				if (method.equals("GET") &&
						path.equals("/items")) {

					List<VectorItem> items = db.all();

					StringBuilder ss = new StringBuilder("[");

					for (int i = 0; i < items.size(); i++) {
						if (i > 0)
							ss.append(',');

						VectorItem v = items.get(i);

						ss.append("{")
								.append("\"id\":").append(v.id)
								.append(",\"metadata\":")
								.append(json(v.metadata))
								.append(",\"category\":")
								.append(json(v.category))
								.append(",\"embedding\":")
								.append(json(v.emb))
								.append("}");
					}

					ss.append("]");

					response(ex, 200, ss.toString());
					return;
				}

				if (method.equals("GET") &&
						path.equals("/search")) {

					Map<String, String> q = queryParams(
							ex.getRequestURI()
									.getRawQuery());

					float[] vector = parseVec(
							q.getOrDefault("v", ""));

					if (vector.length != DIMS) {
						response(
								ex,
								400,
								"{\"error\":\"need " +
										DIMS +
										"D vector\"}");
						return;
					}

					int k = parseInt(
							q.getOrDefault("k", "5"),
							5);

					String metric = q.getOrDefault(
							"metric",
							"cosine");

					String algo = q.getOrDefault(
							"algo",
							"hnsw");

					response(
							ex,
							200,
							vectorResultsJson(
									db.search(
											vector,
											k,
											metric,
											algo)));

					return;
				}

				if (method.equals("POST") &&
						path.equals("/insert")) {

					String b = body(ex);

					StringHolder meta = new StringHolder();

					StringHolder cat = new StringHolder();

					FloatArrayHolder emb = new FloatArrayHolder();

					if (!parseBody(
							b,
							meta,
							cat,
							emb) || emb.value.length != DIMS) {

						response(
								ex,
								400,
								"{\"error\":\"invalid body\"}");

						return;
					}

					int id = db.insert(
							meta.value,
							cat.value,
							emb.value,
							VectorDBApplication::cosine);

					response(
							ex,
							200,
							"{\"id\":" +
									id +
									"}");

					return;
				}

				if (method.equals("DELETE") &&
						path.matches("/delete/[0-9]+")) {

					int id = Integer.parseInt(
							path.substring(
									"/delete/".length()));

					boolean ok = db.remove(id);

					response(
							ex,
							200,
							"{\"ok\":" +
									ok +
									"}");

					return;
				}

				if (method.equals("GET") &&
						path.equals("/benchmark")) {

					Map<String, String> q = queryParams(
							ex.getRequestURI()
									.getRawQuery());

					float[] vector = parseVec(
							q.getOrDefault("v", ""));

					if (vector.length != DIMS) {
						response(
								ex,
								400,
								"{\"error\":\"need " +
										DIMS +
										"D vector\"}");
						return;
					}

					int k = parseInt(
							q.getOrDefault(
									"k",
									"5"),
							5);

					String metric = q.getOrDefault(
							"metric",
							"cosine");

					VectorDB.BenchOut b = db.benchmark(
							vector,
							k,
							metric);

					response(
							ex,
							200,
							"{\"bruteforceUs\":" +
									b.bfUs +
									",\"kdtreeUs\":" +
									b.kdUs +
									",\"hnswUs\":" +
									b.hnswUs +
									",\"itemCount\":" +
									b.n +
									"}");

					return;
				}

				if (method.equals("GET") &&
						path.equals("/hnsw-info")) {

					HNSW.GraphInfo g = db.hnswInfo();

					StringBuilder ss = new StringBuilder();

					ss.append("{")
							.append("\"topLayer\":")
							.append(g.topLayer)
							.append(",\"nodeCount\":")
							.append(g.nodeCount)
							.append(",\"nodesPerLayer\":")
							.append(intListJson(
									g.nodesPerLayer))
							.append(",\"edgesPerLayer\":")
							.append(intListJson(
									g.edgesPerLayer))
							.append(",\"nodes\":[");

					for (int i = 0; i < g.nodes.size(); i++) {

						if (i > 0)
							ss.append(',');

						HNSW.NodeView n = g.nodes.get(i);

						ss.append("{")
								.append("\"id\":")
								.append(n.id)
								.append(",\"metadata\":")
								.append(json(n.metadata))
								.append(",\"category\":")
								.append(json(n.category))
								.append(",\"maxLyr\":")
								.append(n.maxLyr)
								.append("}");
					}

					ss.append("],\"edges\":[");

					for (int i = 0; i < g.edges.size(); i++) {

						if (i > 0)
							ss.append(',');

						HNSW.EdgeView e = g.edges.get(i);

						ss.append("{")
								.append("\"src\":")
								.append(e.src)
								.append(",\"dst\":")
								.append(e.dst)
								.append(",\"lyr\":")
								.append(e.lyr)
								.append("}");
					}

					ss.append("]}");

					response(
							ex,
							200,
							ss.toString());

					return;
				}

				if (method.equals("POST") &&
						path.equals("/doc/insert")) {

					String b = body(ex);

					String title = extractString(
							b,
							"title");

					String text = extractString(
							b,
							"text");

					if (title.isEmpty() ||
							text.isEmpty()) {

						response(
								ex,
								400,
								"{\"error\":\"need title and text\"}");

						return;
					}

					List<String> chunks = chunkText(
							text,
							250,
							30);

					List<Integer> ids = new ArrayList<>();

					for (int i = 0; i < chunks.size(); i++) {

						float[] emb = ollama.embedVector(
								chunks.get(i));

						if (emb.length == 0) {
							response(
									ex,
									503,
									"{\"error\":\"Ollama unavailable. Run ollama serve and pull the embedding model.\"}");
							return;
						}

						String chunkTitle = chunks.size() > 1
								? title +
										" [" +
										(i + 1) +
										"/" +
										chunks.size() +
										"]"
								: title;

						ids.add(
								docDB.insert(
										chunkTitle,
										chunks.get(i),
										emb));
					}

					response(
							ex,
							200,
							"{\"ids\":" +
									intListJson(ids) +
									",\"chunks\":" +
									chunks.size() +
									",\"dims\":" +
									docDB.getDims() +
									"}");

					return;
				}

				if (method.equals("DELETE") &&
						path.matches(
								"/doc/delete/[0-9]+")) {

					int id = Integer.parseInt(
							path.substring(
									"/doc/delete/"
											.length()));

					boolean ok = docDB.remove(id);

					response(
							ex,
							200,
							"{\"ok\":" +
									ok +
									"}");

					return;
				}

				if (method.equals("GET") &&
						path.equals("/doc/list")) {

					List<DocItem> docs = docDB.all();

					StringBuilder ss = new StringBuilder("[");

					for (int i = 0; i < docs.size(); i++) {

						if (i > 0)
							ss.append(',');

						DocItem d = docs.get(i);

						String preview = d.text.substring(
								0,
								Math.min(
										120,
										d.text.length()));

						if (d.text.length() > 120)
							preview += "…";

						int words = d.text.trim().isEmpty()
								? 0
								: d.text.trim()
										.split("\\s+").length;

						ss.append("{")
								.append("\"id\":")
								.append(d.id)
								.append(",\"title\":")
								.append(json(d.title))
								.append(",\"preview\":")
								.append(json(preview))
								.append(",\"words\":")
								.append(words)
								.append("}");
					}

					ss.append("]");

					response(
							ex,
							200,
							ss.toString());

					return;
				}

				if (method.equals("POST") &&
						path.equals("/doc/search")) {

					String b = body(ex);

					String question = extractString(
							b,
							"question");

					int k = extractInt(
							b,
							"k",
							3);

					if (question.isEmpty()) {
						response(
								ex,
								400,
								"{\"error\":\"need question\"}");
						return;
					}

					float[] qEmb = ollama.embedVector(
							question);

					if (qEmb.length == 0) {
						response(
								ex,
								503,
								"{\"error\":\"Ollama unavailable\"}");
						return;
					}

					List<DocumentDB.DocHit> hits = docDB.search(
							qEmb,
							k,
							0.7f);

					StringBuilder ss = new StringBuilder(
							"{\"contexts\":[");

					for (int i = 0; i < hits.size(); i++) {

						if (i > 0)
							ss.append(',');

						DocumentDB.DocHit hit = hits.get(i);

						ss.append("{")
								.append("\"id\":")
								.append(hit.item.id)
								.append(",\"title\":")
								.append(json(
										hit.item.title))
								.append(",\"distance\":")
								.append(String.format(
										Locale.US,
										"%.4f",
										hit.distance))
								.append("}");
					}

					ss.append("]}");

					response(
							ex,
							200,
							ss.toString());

					return;
				}

				if (method.equals("POST") &&
						path.equals("/doc/ask")) {

					String b = body(ex);

					String question = extractString(
							b,
							"question");

					int k = extractInt(
							b,
							"k",
							3);

					if (question.isEmpty()) {
						response(
								ex,
								400,
								"{\"error\":\"need question\"}");
						return;
					}

					float[] qEmb = ollama.embedVector(
							question);

					if (qEmb.length == 0) {
						response(
								ex,
								503,
								"{\"error\":\"Ollama unavailable\"}");
						return;
					}

					List<DocumentDB.DocHit> hits = docDB.search(
							qEmb,
							k,
							0.7f);

					StringBuilder context = new StringBuilder();

					for (int i = 0; i < hits.size(); i++) {

						DocumentDB.DocHit hit = hits.get(i);

						context
								.append("[")
								.append(i + 1)
								.append("] ")
								.append(hit.item.title)
								.append(":\n")
								.append(hit.item.text)
								.append("\n\n");
					}

					String prompt = "You are a helpful assistant. " +
							"Answer the user's question directly. " +
							"Use the provided context if it contains relevant information. "
							+
							"If it doesn't, just use your own general knowledge. " +
							"IMPORTANT: Do NOT mention the context, provided text, " +
							"or say things like the context doesn't mention. " +
							"Just answer the question naturally.\n\n" +
							"Context:\n" +
							context +
							"Question: " +
							question +
							"\n\nAnswer:";

					String answer = ollama.generate(prompt);

					StringBuilder ss = new StringBuilder();

					ss.append("{")
							.append("\"answer\":")
							.append(json(answer))
							.append(",\"model\":")
							.append(json(ollama.genModel))
							.append(",\"contexts\":[");

					for (int i = 0; i < hits.size(); i++) {

						if (i > 0)
							ss.append(',');

						DocumentDB.DocHit hit = hits.get(i);

						ss.append("{")
								.append("\"id\":")
								.append(hit.item.id)
								.append(",\"title\":")
								.append(json(
										hit.item.title))
								.append(",\"text\":")
								.append(json(
										hit.item.text))
								.append(",\"distance\":")
								.append(String.format(
										Locale.US,
										"%.4f",
										hit.distance))
								.append("}");
					}

					ss.append("]")
							.append(",\"docCount\":")
							.append(docDB.size())
							.append("}");

					response(
							ex,
							200,
							ss.toString());

					return;
				}

				response(
						ex,
						404,
						"{\"error\":\"not found\"}");

			} catch (Exception e) {
				response(
						ex,
						500,
						"{\"error\":" +
								json(
										e.getMessage() == null
												? "internal error"
												: e.getMessage())
								+
								"}");
			}
		}
	}

	static int parseInt(String s, int def) {
		try {
			return Integer.parseInt(s);
		} catch (Exception e) {
			return def;
		}
	}

	static Map<String, String> queryParams(String query) {
		Map<String, String> result = new HashMap<>();

		if (query == null || query.isEmpty())
			return result;

		for (String pair : query.split("&")) {
			String[] parts = pair.split(
					"=",
					2);

			if (parts.length == 2) {
				try {
					result.put(
							URLDecoder.decode(
									parts[0],
									StandardCharsets.UTF_8),
							URLDecoder.decode(
									parts[1],
									StandardCharsets.UTF_8));
				} catch (Exception ignored) {
				}
			}
		}

		return result;
	}

	static String intListJson(List<Integer> values) {
		StringBuilder sb = new StringBuilder("[");

		for (int i = 0; i < values.size(); i++) {
			if (i > 0)
				sb.append(',');
			sb.append(values.get(i));
		}

		return sb.append(']').toString();
	}

	public static void main(String[] args) throws Exception {
		new ServerApp().start();
	}
}
