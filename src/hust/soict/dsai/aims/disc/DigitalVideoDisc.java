package hust.soict.dsai.aims.disc;

public class DigitalVideoDisc {
	private String title;
	private String category;
	private String director;
	private int length;
	private float cost;
	private int id;
	
	private static int nbDigitalVideoDiscs = 0; 
	

	public DigitalVideoDisc(String title) {
		super();
		this.title = title;
		this.id = nbDigitalVideoDiscs;
		nbDigitalVideoDiscs += 1; 
	}
	
	public DigitalVideoDisc(String title, String category, float cost) {
		super();
		this.title = title;
		this.category = category;
		this.cost = cost;
		this.id = nbDigitalVideoDiscs;
		nbDigitalVideoDiscs += 1;
	}
	public DigitalVideoDisc(String title, String category, String director, float cost) {
		super();
		this.title = title;
		this.category = category;
		this.director = director;
		this.cost = cost;
		this.id = nbDigitalVideoDiscs;
		nbDigitalVideoDiscs += 1;
	}
	public DigitalVideoDisc(String title, String category, String director, int length, float cost) {
		super();
		this.title = title;
		this.category = category;
		this.director = director;
		this.length = length;
		this.cost = cost;
		this.id = nbDigitalVideoDiscs;
		nbDigitalVideoDiscs += 1;
	}
	public String getTitle() {
		return title;
	}
	public String getCategory() {
		return category;
	}
	public String getDirector() {
		return director;
	}
	public int getLength() {
		return length;
	}
	public float getCost() {	
		return cost;
	}
	public int getId() {
		return id;
	}
	
	public void setTitle(String title) {
        this.title = title;
    }

	public String toString() {
	    return "DVD - " + title + " - " + category + " - " + director + 
	           " - " + length + " - " + cost + "$";
	}

	public boolean isMatch(String title) {
	    return this.title.toLowerCase().contains(title.toLowerCase());
	}

}
