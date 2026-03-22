#include "Vocabulary.h"
#include "DescManip.h"
#include "quicklz.h"
#include <sstream>
#include "timers.h"
namespace DBoW3{
// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (int k, int L, WeightingType weighting, ScoringType scoring)
  : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
  m_scoring_object(NULL)
{
  createScoringObject();
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (const std::string &filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (const char *filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (std::istream& stream): m_scoring_object(NULL)
{
  load(stream);
}

// --------------------------------------------------------------------------


void Vocabulary::createScoringObject()
{
  delete m_scoring_object;
  m_scoring_object = NULL;

  switch(m_scoring)
  {
    case L1_NORM:
      m_scoring_object = new L1Scoring;
      break;

    case L2_NORM:
      m_scoring_object = new L2Scoring;
      break;

    case CHI_SQUARE:
      m_scoring_object = new ChiSquareScoring;
      break;

    case KL:
      m_scoring_object = new KLScoring;
      break;

    case BHATTACHARYYA:
      m_scoring_object = new BhattacharyyaScoring;
      break;

    case DOT_PRODUCT:
      m_scoring_object = new DotProductScoring;
      break;

  }
}

// --------------------------------------------------------------------------


void Vocabulary::setScoringType(ScoringType type)
{
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------


void Vocabulary::setWeightingType(WeightingType type)
{
  this->m_weighting = type;
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary(
  const Vocabulary &voc)
  : m_scoring_object(NULL)
{
  *this = voc;
}

// --------------------------------------------------------------------------


Vocabulary::~Vocabulary()
{
  delete m_scoring_object;
}

// --------------------------------------------------------------------------


Vocabulary&
Vocabulary::operator=
  (const Vocabulary &voc)
{
  this->m_k = voc.m_k;
  this->m_L = voc.m_L;
  this->m_scoring = voc.m_scoring;
  this->m_weighting = voc.m_weighting;

  this->createScoringObject();

  this->m_nodes.clear();
  this->m_words.clear();

  this->m_nodes = voc.m_nodes;
  this->createWords();

  return *this;
}



void Vocabulary::create(
  const std::vector< cv::Mat > &training_features)
{
    std::vector<std::vector<cv::Mat> > vtf(training_features.size());
    for(size_t i=0;i<training_features.size();i++){
        vtf[i].resize(training_features[i].rows);
        for(int r=0;r<training_features[i].rows;r++)
            vtf[i][r]=training_features[i].rowRange(r,r+1);
    }
    create(vtf);

}

void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features)
{
  m_nodes.clear();
  m_words.clear();

  // expected_nodes = Sum_{i=0..L} ( k^i )
    int expected_nodes =
        (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree


  std::vector<cv::Mat> features;
  getFeatures(training_features, features);


  // create root
  m_nodes.push_back(Node(0)); // root

  // create the tree
  HKmeansStep(0, features, 1);

  // create the words
  createWords();

  // and set the weight of each node of the tree
  setNodeWeights(training_features);

}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features,
  int k, int L)
{
  m_k = k;
  m_L = L;

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<cv::Mat> > &training_features,
  int k, int L, WeightingType weighting, ScoringType scoring)
{
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::getFeatures(
  const std::vector<std::vector<cv::Mat> > &training_features,
  std::vector<cv::Mat> &features) const
{
  features.resize(0);
  for(size_t i=0;i<training_features.size();i++)
      for(size_t j=0;j<training_features[i].size();j++)
              features.push_back(training_features[i][j]);
}

// --------------------------------------------------------------------------


void Vocabulary::HKmeansStep(NodeId parent_id,
                             const std::vector<cv::Mat> &descriptors, int current_level)
{

    if(descriptors.empty()) return;

    // features associated to each cluster
    std::vector<cv::Mat> clusters;
    std::vector<std::vector<unsigned int> > groups; // groups[i] = [j1, j2, ...]
    // j1, j2, ... indices of descriptors associated to cluster i

    clusters.reserve(m_k);
    groups.reserve(m_k);


    if((int)descriptors.size() <= m_k)
    {
        // trivial case: one cluster per feature
        groups.resize(descriptors.size());

        for(unsigned int i = 0; i < descriptors.size(); i++)
        {
            groups[i].push_back(i);
            clusters.push_back(descriptors[i]);
        }
    }
    else
    {
        // select clusters and groups with kmeans

        bool first_time = true;
        bool goon = true;

        // to check if clusters move after iterations
        std::vector<int> last_association, current_association;

        while(goon)
        {
            // 1. Calculate clusters

            if(first_time)
            {
                // random sample
                initiateClusters(descriptors, clusters);
            }
            else
            {
                // calculate cluster centres

                for(unsigned int c = 0; c < clusters.size(); ++c)
                {
                    std::vector<cv::Mat> cluster_descriptors;
                    cluster_descriptors.reserve(groups[c].size());
                    std::vector<unsigned int>::const_iterator vit;
                    for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
                    {
                        cluster_descriptors.push_back(descriptors[*vit]);
                    }

                    DescManip::meanValue(cluster_descriptors, clusters[c]);
                }

            } // if(!first_time)

            // 2. Associate features with clusters

            // calculate distances to cluster centers
            groups.clear();
            groups.resize(clusters.size(), std::vector<unsigned int>());
            current_association.resize(descriptors.size());

            //assoc.clear();

            //unsigned int d = 0;
            for(auto  fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)
            {
                double best_dist = DescManip::distance((*fit), clusters[0]);
                unsigned int icluster = 0;

                for(unsigned int c = 1; c < clusters.size(); ++c)
                {
                    double dist = DescManip::distance((*fit), clusters[c]);
                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        icluster = c;
                    }
                }

                //assoc.ref<unsigned char>(icluster, d) = 1;

                groups[icluster].push_back(fit - descriptors.begin());
                current_association[ fit - descriptors.begin() ] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them

            // 3. check convergence
            if(first_time)
            {
                first_time = false;
            }
            else
            {
                //goon = !eqUChar(last_assoc, assoc);

                goon = false;
                for(unsigned int i = 0; i < current_association.size(); i++)
                {
                    if(current_association[i] != last_association[i]){
                        goon = true;
                        break;
                    }
                }
            }

            if(goon)
            {
                // copy last feature-cluster association
                last_association = current_association;
                //last_assoc = assoc.clone();
            }

        } // while(goon)

    } // if must run kmeans

    // create nodes
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
        NodeId id = m_nodes.size();
        m_nodes.push_back(Node(id));
        m_nodes.back().descriptor = clusters[i];
        m_nodes.back().parent = parent_id;
        m_nodes[parent_id].children.push_back(id);
    }

    // go on with the next level
    if(current_level < m_L)
    {
        // iterate again with the resulting clusters
        const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
        for(unsigned int i = 0; i < clusters.size(); ++i)
        {
            NodeId id = children_ids[i];

            std::vector<cv::Mat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)
            {
                child_features.push_back(descriptors[*vit]);
            }

            if(child_features.size() > 1)
            {
                HKmeansStep(id, child_features, current_level + 1);
            }
        }
    }
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClusters
  (const std::vector<cv::Mat> &descriptors,
   std::vector<cv::Mat> &clusters) const
{
  initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClustersKMpp(
  const std::vector<cv::Mat> &pfeatures,
    std::vector<cv::Mat> &clusters) const
{
  // Implements kmeans++ seeding algorithm
  // Algorithm:
  // 1. Choose one center uniformly at random from among the data points.
  // 2. For each data point x, compute D(x), the distance between x and the nearest
  //    center that has already been chosen.
  // 3. Add one new data point as a center. Each point x is chosen with probability
  //    proportional to D(x)^2.
  // 4. Repeat Steps 2 and 3 until k centers have been chosen.
  // 5. Now that the initial centers have been chosen, proceed using standard k-means
  //    clustering.


//  DUtils::Random::SeedRandOnce();

  clusters.resize(0);
  clusters.reserve(m_k);
  std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

  // 1.

  int ifeature = rand()% pfeatures.size();//DUtils::Random::RandomInt(0, pfeatures.size()-1);

  // create first cluster
  clusters.push_back(pfeatures[ifeature]);

  // compute the initial distances
   std::vector<double>::iterator dit;
  dit = min_dists.begin();
  for(auto fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
  {
    *dit = DescManip::distance((*fit), clusters.back());
  }

  while((int)clusters.size() < m_k)
  {
    // 2.
    dit = min_dists.begin();
    for(auto  fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
    {
      if(*dit > 0)
      {
        double dist = DescManip::distance((*fit), clusters.back());
        if(dist < *dit) *dit = dist;
      }
    }

    // 3.
    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

    if(dist_sum > 0)
    {
      double cut_d;
      do
      {

        cut_d = (double(rand())/ double(RAND_MAX))* dist_sum;
      } while(cut_d == 0.0);

      double d_up_now = 0;
      for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
      {
        d_up_now += *dit;
        if(d_up_now >= cut_d) break;
      }

      if(dit == min_dists.end())
        ifeature = pfeatures.size()-1;
      else
        ifeature = dit - min_dists.begin();


      clusters.push_back(pfeatures[ifeature]);
    } // if dist_sum > 0
    else
      break;

  } // while(used_clusters < m_k)

}

// --------------------------------------------------------------------------


void Vocabulary::createWords()
{
  m_words.resize(0);

  if(!m_nodes.empty())
  {
    m_words.reserve( (int)pow((double)m_k, (double)m_L) );


    auto  nit = m_nodes.begin(); // ignore root
    for(++nit; nit != m_nodes.end(); ++nit)
    {
      if(nit->isLeaf())
      {
        nit->word_id = m_words.size();
        m_words.push_back( &(*nit) );
      }
    }
  }
}

// --------------------------------------------------------------------------


void Vocabulary::setNodeWeights
  (const std::vector<std::vector<cv::Mat> > &training_features)
{
  const unsigned int NWords = m_words.size();
  const unsigned int NDocs = training_features.size();

  if(m_weighting == TF || m_weighting == BINARY)
  {
    // idf part must be 1 always
    for(unsigned int i = 0; i < NWords; i++)
      m_words[i]->weight = 1;
  }
  else if(m_weighting == IDF || m_weighting == TF_IDF)
  {
    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);


    for(auto mit = training_features.begin(); mit != training_features.end(); ++mit)
    {
      fill(counted.begin(), counted.end(), false);

      for(auto fit = mit->begin(); fit < mit->end(); ++fit)
      {
        WordId word_id;
        transform(*fit, word_id);

        if(!counted[word_id])
        {
          Ni[word_id]++;
          counted[word_id] = true;
        }
      }
    }

    // set ln(N/Ni)
    for(unsigned int i = 0; i < NWords; i++)
    {
      if(Ni[i] > 0)
      {
        m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
      }// else // This cannot occur if using kmeans++
    }

  }

}

// --------------------------------------------------------------------------






// --------------------------------------------------------------------------


float Vocabulary::getEffectiveLevels() const
{
  long sum = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    const Node *p = *wit;

    for(; p->id != 0; sum++) p = &m_nodes[p->parent];
  }

  return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------


cv::Mat Vocabulary::getWord(WordId wid) const
{
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------


WordValue Vocabulary::getWordWeight(WordId wid) const
{
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------


WordId Vocabulary::transform
  (const cv::Mat& feature) const
{
  if(empty())
  {
    return 0;
  }

  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------

void Vocabulary::transform(
        const cv::Mat& features, BowVector &v) const
{
    //    std::vector<cv::Mat> vf(features.rows);
    //    for(int r=0;r<features.rows;r++) vf[r]=features.rowRange(r,r+1);
    //    transform(vf,v);



    v.clear();

    if(empty())
    {
        return;
    }

    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);


    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is the idf value if TF_IDF, 1 if TF
            transform(features.row(r), id, w);
            // not stopped
            if(w > 0)  v.addWeight(id, w);
        }

        if(!v.empty() && !must)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                vit->second /= nd;
        }

    }
    else // IDF || BINARY
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is idf if IDF, or 1 if BINARY

            transform(features.row(r), id, w);

            // not stopped
            if(w > 0) v.addIfNotExist(id, w);

        } // if add_features
    } // if m_weighting == ...

    if(must) v.normalize(norm);

}



void Vocabulary::transform(
  const std::vector<cv::Mat>& features, BowVector &v) const
{
  v.clear();

  if(empty())
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w);

      // not stopped
      if(w > 0) v.addWeight(id, w);
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w);

      // not stopped
      if(w > 0) v.addIfNotExist(id, w);

    } // if add_features
  } // if m_weighting == ...

  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------


void Vocabulary::transform(
  const std::vector<cv::Mat>& features,
  BowVector &v, FeatureVector &fv, int levelsup) const
{
  v.clear();
  fv.clear();

  if(empty()) // safe for subclasses
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        v.addWeight(id, w);
        fv.addFeature(nid, i_feature);
      }
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        v.addIfNotExist(id, w);
        fv.addFeature(nid, i_feature);
      }
    }
  } // if m_weighting == ...

  if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------


// --------------------------------------------------------------------------


void Vocabulary::transform
  (const cv::Mat &feature, WordId &id) const
{
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------


void Vocabulary::transform(const cv::Mat &feature,
  WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{
  // propagate the feature down the tree


  // level at which the node must be stored in nid, if given
  const int nid_level = m_L - levelsup;
  if(nid_level <= 0 && nid != NULL) *nid = 0; // root

  NodeId final_id = 0; // root
  int current_level = 0;

  do
  {
    ++current_level;
    auto const  &nodes = m_nodes[final_id].children;
    double best_d = std::numeric_limits<double>::max();
//    DescManip::distance(feature, m_nodes[final_id].descriptor);

    for(const auto  &id:nodes)
    {
      double d = DescManip::distance(feature, m_nodes[id].descriptor);
      if(d < best_d)
      {
        best_d = d;
        final_id = id;
      }
    }

    if(nid != NULL && current_level == nid_level)
      *nid = final_id;

  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}



void Vocabulary::transform(const cv::Mat &feature,
  WordId &word_id, WordValue &weight ) const
{
  // propagate the feature down the tree


  // level at which the node must be stored in nid, if given

  NodeId final_id = 0; // root
//maximum speed by computing here distance and avoid calling to DescManip::distance

  //binary descriptor
 // int ntimes=0;
  if (feature.type()==CV_8U){
      do
      {
          auto const  &nodes = m_nodes[final_id].children;
          uint64_t best_d = std::numeric_limits<uint64_t>::max();
          int idx=0,bestidx=0;
           for(const auto  &id:nodes)
          {
              //compute distance
             //  std::cout<<idx<< " "<<id<<" "<< m_nodes[id].descriptor<<std::endl;
              uint64_t dist= DescManip::distance_8uc1(feature, m_nodes[id].descriptor);
              if(dist < best_d)
              {
                  best_d = dist;
                  final_id = id;
                  bestidx=idx;
              }
              idx++;
          }
        // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<" "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
      } while( !m_nodes[final_id].isLeaf() );
   }
  else
  {
	  do
	  {
		  auto const  &nodes = m_nodes[final_id].children;
		  uint64_t best_d = std::numeric_limits<uint64_t>::max();
		  int idx = 0, bestidx = 0;
		  for (const auto &id : nodes)
		  {
			  //compute distance
			  //  std::cout<<idx<< " "<<id<<" "<< m_nodes[id].descriptor<<std::endl;
			  uint64_t dist = DescManip::distance(feature, m_nodes[id].descriptor);
			  //std::cout << id << " " << dist << " " << best_d << std::endl;
			  if (dist < best_d)
			  {
				  best_d = dist;
				  final_id = id;
				  bestidx = idx;
			  }
			  idx++;
		  }
		  // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<" "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
	  } while (!m_nodes[final_id].isLeaf());
  }
//      uint64_t ret=0;
//      const uchar *pb = b.ptr<uchar>();
//      for(int i=0;i<a.cols;i++,pa++,pb++){
//          uchar v=(*pa)^(*pb);
//#ifdef __GNUG__
//          ret+=__builtin_popcount(v);//only in g++
//#else

//          ret+=v& (1<<0);
//          ret+=v& (1<<1);
//          ret+=v& (1<<2);
//          ret+=v& (1<<3);
//          ret+=v& (1<<4);
//          ret+=v& (1<<5);
//          ret+=v& (1<<6);
//          ret+=v& (1<<7);
//#endif
//  }
//      return ret;
//  }
//  else{
//      double sqd = 0.;
//      assert(a.type()==CV_32F);
//      assert(a.rows==1);
//      const float *a_ptr=a.ptr<float>(0);
//      const float *b_ptr=b.ptr<float>(0);
//      for(int i = 0; i < a.cols; i ++)
//          sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
//      return sqd;
//  }


//  do
//  {
//    auto const  &nodes = m_nodes[final_id].children;
//    double best_d = std::numeric_limits<double>::max();

//    for(const auto  &id:nodes)
//    {
//      double d = DescManip::distance(feature, m_nodes[id].descriptor);
//      if(d < best_d)
//      {
//        best_d = d;
//        final_id = id;
//      }
//    }
//  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}
// --------------------------------------------------------------------------

NodeId Vocabulary::getParentNode
  (WordId wid, int levelsup) const
{
  NodeId ret = m_words[wid]->id; // node id
  while(levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------


void Vocabulary::getWordsFromNode
  (NodeId nid, std::vector<WordId> &words) const
{
  words.clear();

  if(m_nodes[nid].isLeaf())
  {
    words.push_back(m_nodes[nid].word_id);
  }
  else
  {
    words.reserve(m_k); // ^1, ^2, ...

    std::vector<NodeId> parents;
    parents.push_back(nid);

    while(!parents.empty())
    {
      NodeId parentid = parents.back();
      parents.pop_back();

      const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
      std::vector<NodeId>::const_iterator cit;

      for(cit = child_ids.begin(); cit != child_ids.end(); ++cit)
      {
        const Node &child_node = m_nodes[*cit];

        if(child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(*cit);

      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------


int Vocabulary::stopWords(double minWeight)
{
  int c = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    if((*wit)->weight < minWeight)
    {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------


void Vocabulary::save(const std::string &filename,  bool binary_compressed) const
{

    if ( filename.find(".yml")==std::string::npos){
        std::ofstream file_out(filename,std::ios::binary);
        if (!file_out) throw std::runtime_error("Vocabulary::saveBinary Could not open file :"+filename+" for writing");
        toStream(file_out,binary_compressed);
    }
    else{
        cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
        if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
        save(fs);
    }
}

// --------------------------------------------------------------------------


// void Vocabulary::load(const std::string &filename)
// {
//     //check first if it is a binary file
//     std::ifstream ifile(filename,std::ios::binary);
//     if (!ifile) throw std::runtime_error("Vocabulary::load Could not open file :"+filename+" for reading");
//     if(!load(ifile)) {
//         if ( filename.find(".txt")!=std::string::npos) {
// 	    load_fromtxt(filename);
// 	} else {
// 	    cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
// 	    if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
// 	    load(fs);
// 	}
//     }
// }


// 修改load方法，避开OpenCV FileStorage
void Vocabulary::load(const std::string &filename)
{
  std::cout << " 使用新方法"<< std::endl;
  // 检查文件大小
  std::ifstream file_check(filename, std::ios::binary | std::ios::ate);
  if (!file_check)
    throw std::runtime_error("Vocabulary::load Could not open file :" + filename + " for reading");

  std::streamsize file_size = file_check.tellg();
  file_check.close();

  // 如果文件大于100MB，使用流式读取
  const std::streamsize LARGE_FILE_THRESHOLD = 100 * 1024 * 1024; // 100MB

  if (file_size > LARGE_FILE_THRESHOLD)
  {
    std::cout << "检测到大文件 (" << file_size / (1024 * 1024) << " MB)，使用流式读取..." << std::endl;
    load_large_file_streaming(filename);
    return;
  }

  // 小文件：先尝试二进制，失败后用YAML文本方式
  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile)
    throw std::runtime_error("Vocabulary::load Could not open file :" + filename + " for reading");

  if (!load(ifile))
  {
    if (filename.find(".txt") != std::string::npos)
    {
      load_fromtxt(filename);
    }
    else
    {
      // 避开FileStorage，直接解析YAML文本
      std::cout << "使用YAML文本解析..." << std::endl;
      load_from_yaml_text(filename);
    }
  }
}

// 完全替换的load_from_yaml_text方法
void Vocabulary::load_from_yaml_text(const std::string &filename)
{
  std::cout << "开始解析YAML词典文件: " << filename << std::endl;

  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("无法打开文件: " + filename);

  // 清理现有数据
  clear();

  std::string line;
  std::string current_section;

  // 第一阶段：解析头部信息
  std::cout << "解析头部信息..." << std::endl;
  while (std::getline(file, line))
  {
    line = trim(line);

    if (line.find("k:") != std::string::npos)
    {
      m_k = std::stoi(extract_value(line));
      std::cout << "k = " << m_k << std::endl;
    }
    else if (line.find("L:") != std::string::npos)
    {
      m_L = std::stoi(extract_value(line));
      std::cout << "L = " << m_L << std::endl;
    }
    else if (line == "nodes:")
    {
      current_section = "nodes";
      std::cout << "开始解析节点..." << std::endl;
      break;
    }
  }

  if (m_k <= 0 || m_L <= 0)
  {
    throw std::runtime_error("无效的词典参数: k=" + std::to_string(m_k) +
                             ", L=" + std::to_string(m_L));
  }

  // 第二阶段：临时存储节点数据
  std::vector<TempNodeData> temp_nodes;
  TempNodeData current_temp_node;
  bool reading_node = false;
  int nodes_count = 0;

  while (std::getline(file, line))
  {
    line = trim(line);

    if (line == "words:")
    {
      if (reading_node)
      {
        temp_nodes.push_back(current_temp_node);
        nodes_count++;
      }
      std::cout << "节点解析完成，共 " << nodes_count << " 个节点" << std::endl;
      current_section = "words";
      break;
    }

    if (current_section == "nodes")
    {
      if (line.find("- nodeId:") != std::string::npos)
      {
        if (reading_node)
        {
          temp_nodes.push_back(current_temp_node);
          nodes_count++;
        }
        current_temp_node = TempNodeData();
        current_temp_node.id = std::stoi(extract_value(line));
        reading_node = true;
      }
      else if (line.find("parentId:") != std::string::npos && reading_node)
      {
        current_temp_node.parent = std::stoi(extract_value(line));
      }
      else if (line.find("weight:") != std::string::npos && reading_node)
      {
        current_temp_node.weight = std::stof(extract_value(line));
      }
      else if (line.find("descriptor:") != std::string::npos && reading_node)
      {
        std::string desc_str = extract_descriptor(line);
        current_temp_node.descriptor = parse_descriptor_string(desc_str);
      }
    }
  }

  // 第三阶段：解析单词映射
  std::cout << "开始解析单词映射..." << std::endl;
  std::vector<WordMapping> word_mappings;
  int word_id = -1;
  int node_id = -1;
  int words_count = 0;

  while (std::getline(file, line))
  {
    line = trim(line);

    if (line.find("- wordId:") != std::string::npos)
    {
      word_id = std::stoi(extract_value(line));
    }
    else if (line.find("nodeId:") != std::string::npos)
    {
      node_id = std::stoi(extract_value(line));

      if (word_id >= 0 && node_id >= 0)
      {
        word_mappings.push_back({word_id, node_id});
        words_count++;
      }
    }
  }

  std::cout << "单词映射解析完成，共 " << words_count << " 个单词" << std::endl;

  // 第四阶段：构建完整的词典结构
  build_vocabulary_from_temp_data(temp_nodes, word_mappings);

  // 第五阶段：验证和完善结构
  rebuild_node_relationships();
  validate_vocabulary_structure();

  std::cout << "词典加载完成! 节点数: " << m_nodes.size()
            << ", 单词数: " << m_words.size() << std::endl;
}

// 新增：从临时数据构建词典结构
void Vocabulary::build_vocabulary_from_temp_data(const std::vector<TempNodeData> &temp_nodes,
                                                 const std::vector<WordMapping> &word_mappings)
{
  std::cout << "构建词典结构..." << std::endl;

  if (temp_nodes.empty())
  {
    throw std::runtime_error("没有找到有效的节点数据");
  }

  // 找到最大节点ID，确定需要的空间
  int max_node_id = 0;
  for (const auto &temp_node : temp_nodes)
  {
    max_node_id = std::max(max_node_id, temp_node.id);
  }

  std::cout << "最大节点ID: " << max_node_id << std::endl;

  // 预分配足够的空间
  m_nodes.clear();
  m_nodes.resize(max_node_id + 1);

  // 初始化所有节点
  for (int i = 0; i <= max_node_id; ++i)
  {
    m_nodes[i].id = i;
    m_nodes[i].parent = -1;
    m_nodes[i].children.clear();
    m_nodes[i].weight = 0.0;
    m_nodes[i].word_id = -1;
    m_nodes[i].descriptor = cv::Mat();
  }

  // 填充节点数据
  int valid_nodes = 0;
  for (const auto &temp_node : temp_nodes)
  {
    if (temp_node.id >= 0 && temp_node.id < m_nodes.size())
    {
      m_nodes[temp_node.id].id = temp_node.id;
      m_nodes[temp_node.id].parent = temp_node.parent;
      m_nodes[temp_node.id].weight = temp_node.weight;

      if (!temp_node.descriptor.empty())
      {
        m_nodes[temp_node.id].descriptor = temp_node.descriptor.clone();
      }
      valid_nodes++;
    }
  }

  std::cout << "有效节点数: " << valid_nodes << "/" << temp_nodes.size() << std::endl;

  // 设置单词映射
  setup_word_mappings(word_mappings);
}

// 新增：设置单词映射
void Vocabulary::setup_word_mappings(const std::vector<WordMapping> &word_mappings)
{
  std::cout << "设置单词映射..." << std::endl;

  if (word_mappings.empty())
  {
    std::cout << "警告: 没有找到单词映射数据" << std::endl;
    return;
  }

  // 找到最大word_id
  int max_word_id = 0;
  for (const auto &mapping : word_mappings)
  {
    max_word_id = std::max(max_word_id, mapping.word_id);
  }

  // 分配words数组
  m_words.clear();
  m_words.resize(max_word_id + 1, nullptr);

  // 设置映射关系
  int valid_mappings = 0;
  for (const auto &mapping : word_mappings)
  {
    int word_id = mapping.word_id;
    int node_id = mapping.node_id;

    if (word_id >= 0 && word_id < m_words.size() &&
        node_id >= 0 && node_id < m_nodes.size())
    {

      m_nodes[node_id].word_id = word_id;
      m_words[word_id] = &m_nodes[node_id];
      valid_mappings++;
    }
  }

  std::cout << "有效单词映射: " << valid_mappings << "/" << word_mappings.size() << std::endl;
}

// 改进的重建节点关系方法
void Vocabulary::rebuild_node_relationships()
{
  std::cout << "重建节点关系..." << std::endl;

  // 清理所有children
  for (auto &node : m_nodes)
  {
    node.children.clear();
  }

  // 重建父子关系
  int relationships_built = 0;
  for (size_t i = 0; i < m_nodes.size(); ++i)
  {
    int parent_id = m_nodes[i].parent;
    if (parent_id >= 0 && parent_id < static_cast<int>(m_nodes.size()))
    {
      m_nodes[parent_id].children.push_back(static_cast<int>(i));
      relationships_built++;
    }
  }

  std::cout << "构建了 " << relationships_built << " 个父子关系" << std::endl;
}

// 新增：验证词典结构
void Vocabulary::validate_vocabulary_structure()
{
  std::cout << "验证词典结构..." << std::endl;

  int valid_nodes = 0;
  int leaf_nodes = 0;
  int nodes_with_descriptors = 0;

  for (size_t i = 0; i < m_nodes.size(); ++i)
  {
    if (m_nodes[i].id >= 0)
    {
      valid_nodes++;

      if (!m_nodes[i].descriptor.empty())
      {
        nodes_with_descriptors++;
      }

      if (m_nodes[i].children.empty())
      {
        leaf_nodes++;
      }
    }
  }

  // 检查words数组的有效性
  int valid_words = 0;
  int null_words = 0;
  for (size_t i = 0; i < m_words.size(); ++i)
  {
    if (m_words[i] != nullptr)
    {
      if (!m_words[i]->descriptor.empty())
      {
        valid_words++;
      }
    }
    else
    {
      null_words++;
    }
  }

  std::cout << "结构验证结果:" << std::endl;
  std::cout << "  有效节点: " << valid_nodes << "/" << m_nodes.size() << std::endl;
  std::cout << "  叶子节点: " << leaf_nodes << std::endl;
  std::cout << "  有描述符的节点: " << nodes_with_descriptors << std::endl;
  std::cout << "  有效单词: " << valid_words << "/" << m_words.size() << std::endl;
  std::cout << "  空单词指针: " << null_words << std::endl;

  // 基本完整性检查
  if (valid_nodes == 0)
  {
    throw std::runtime_error("词典结构验证失败: 没有有效节点");
  }

  if (valid_words == 0)
  {
    std::cout << "警告: 没有有效的单词映射" << std::endl;
  }
}


// 完全重写的解析描述符字符串方法
cv::Mat Vocabulary::parse_descriptor_string(const std::string &desc_str)
{
  if (desc_str.empty())
    return cv::Mat();

  std::cout << "解析描述符: " << desc_str.substr(0, 50) << "..." << std::endl;

  // 检测描述符格式
  if (desc_str.find("dbw3") == 0)
  {
    // DBoW3二进制格式
    return parse_dbow3_binary_descriptor(desc_str);
  }
  else if (desc_str.find('[') != std::string::npos && desc_str.find(',') != std::string::npos)
  {
    // 标准浮点数组格式 [1.0, 2.0, 3.0]
    return parse_float_array_descriptor(desc_str);
  }
  else if (desc_str.find_first_not_of("01") == std::string::npos)
  {
    // 纯二进制字符串
    return parse_binary_string_descriptor(desc_str);
  }
  else
  {
    // 尝试其他格式
    return parse_generic_descriptor(desc_str);
  }
}

// 新增：解析DBoW3二进制格式描述符
cv::Mat Vocabulary::parse_dbow3_binary_descriptor(const std::string &desc_str)
{
  std::cout << "检测到DBoW3二进制格式" << std::endl;

  // DBoW3格式: "dbw3" + 尺寸信息 + 二进制数据
  if (desc_str.length() < 10)
  {
    std::cout << "DBoW3描述符太短" << std::endl;
    return cv::Mat();
  }

  try
  {
    // 提取头部信息
    std::string header = desc_str.substr(0, 4); // "dbw3"
    if (header != "dbw3")
    {
      std::cout << "无效的DBoW3头部: " << header << std::endl;
      return cv::Mat();
    }

    // 提取尺寸信息 (假设接下来6位是尺寸)
    std::string size_str = desc_str.substr(4, 6); // "025600" -> 256
    int descriptor_size = 0;

    // 解析尺寸
    try
    {
      descriptor_size = std::stoi(size_str);
    }
    catch (...)
    {
      // 如果尺寸解析失败，使用常见的ORB描述符尺寸
      descriptor_size = 256; // ORB描述符通常是32字节=256位
    }

    std::cout << "描述符尺寸: " << descriptor_size << " 位" << std::endl;

    // 提取二进制数据部分
    std::string binary_data = desc_str.substr(10); // 跳过头部

    // 转换二进制字符串为字节数组
    return convert_binary_string_to_mat(binary_data, descriptor_size);
  }
  catch (const std::exception &e)
  {
    std::cout << "DBoW3描述符解析失败: " << e.what() << std::endl;
    return cv::Mat();
  }
}

// 新增：转换二进制字符串为Mat
cv::Mat Vocabulary::convert_binary_string_to_mat(const std::string &binary_str, int expected_bits)
{
  if (binary_str.empty())
  {
    return cv::Mat();
  }

  // 计算需要的字节数
  int bytes_needed = (expected_bits + 7) / 8; // 向上取整
  int actual_bits = binary_str.length();

  std::cout << "二进制数据长度: " << actual_bits << " 位, 需要: " << expected_bits << " 位" << std::endl;

  // 如果二进制字符串太短，补零
  std::string padded_binary = binary_str;
  while (padded_binary.length() < expected_bits)
  {
    padded_binary += "0";
  }

  // 如果太长，截断
  if (padded_binary.length() > expected_bits)
  {
    padded_binary = padded_binary.substr(0, expected_bits);
  }

  // 转换为字节数组
  std::vector<uchar> bytes;

  for (size_t i = 0; i < padded_binary.length(); i += 8)
  {
    std::string byte_str = padded_binary.substr(i, 8);

    // 如果不足8位，补零
    while (byte_str.length() < 8)
    {
      byte_str += "0";
    }

    // 转换为字节值
    uchar byte_val = 0;
    for (int j = 0; j < 8; ++j)
    {
      if (j < byte_str.length() && byte_str[j] == '1')
      {
        byte_val |= (1 << (7 - j));
      }
    }
    bytes.push_back(byte_val);
  }

  if (bytes.empty())
  {
    return cv::Mat();
  }

  // 创建OpenCV Mat (使用CV_8U类型，这是ORB描述符的标准类型)
  cv::Mat descriptor(1, bytes.size(), CV_8U);
  for (size_t i = 0; i < bytes.size(); ++i)
  {
    descriptor.at<uchar>(0, i) = bytes[i];
  }

  std::cout << "成功创建描述符: " << descriptor.rows << "x" << descriptor.cols
            << ", 类型: " << descriptor.type() << std::endl;

  return descriptor;
}

// 新增：解析标准浮点数组格式
cv::Mat Vocabulary::parse_float_array_descriptor(const std::string &desc_str)
{
  std::cout << "解析浮点数组格式" << std::endl;

  std::vector<float> values;
  std::string cleaned_str = desc_str;

  // 移除所有括号
  cleaned_str.erase(std::remove_if(cleaned_str.begin(), cleaned_str.end(),
                                   [](char c)
                                   { return c == '[' || c == ']'; }),
                    cleaned_str.end());

  std::stringstream ss(cleaned_str);
  std::string token;

  // 按逗号分割
  while (std::getline(ss, token, ','))
  {
    token = trim(token);
    if (!token.empty())
    {
      try
      {
        float value = std::stof(token);
        values.push_back(value);
      }
      catch (const std::exception &e)
      {
        std::cout << "警告: 无法解析浮点值: " << token << std::endl;
      }
    }
  }

  if (values.empty())
  {
    return cv::Mat();
  }

  cv::Mat descriptor(1, static_cast<int>(values.size()), CV_32F);
  for (size_t i = 0; i < values.size(); ++i)
  {
    descriptor.at<float>(0, static_cast<int>(i)) = values[i];
  }

  return descriptor;
}

// 新增：解析纯二进制字符串
cv::Mat Vocabulary::parse_binary_string_descriptor(const std::string &desc_str)
{
  std::cout << "解析纯二进制字符串" << std::endl;
  return convert_binary_string_to_mat(desc_str, desc_str.length());
}

// 新增：通用描述符解析器
cv::Mat Vocabulary::parse_generic_descriptor(const std::string &desc_str)
{
  std::cout << "尝试通用解析" << std::endl;

  // 尝试按空格分割
  std::vector<float> values;
  std::stringstream ss(desc_str);
  std::string token;

  while (ss >> token)
  {
    try
    {
      float value = std::stof(token);
      values.push_back(value);
    }
    catch (const std::exception &)
    {
      // 忽略无法解析的token
    }
  }

  if (values.empty())
  {
    std::cout << "无法解析描述符，返回空Mat" << std::endl;
    return cv::Mat();
  }

  cv::Mat descriptor(1, static_cast<int>(values.size()), CV_32F);
  for (size_t i = 0; i < values.size(); ++i)
  {
    descriptor.at<float>(0, static_cast<int>(i)) = values[i];
  }

  return descriptor;
}




// 保持不变的流式读取方法
void Vocabulary::load_large_file_streaming(const std::string &filename)
{
  load_from_yaml_text(filename);
}

// 修复的清理方法
void Vocabulary::clear()
{
  // 清理scoring object
  if (m_scoring_object)
  {
    delete m_scoring_object;
    m_scoring_object = nullptr;
  }

  // 清理容器
  m_nodes.clear();
  m_words.clear();

  // 重置参数 - 这些不能注释掉！
  m_k = 0;
  m_L = 0;
}

// 保持不变的辅助方法
std::string Vocabulary::trim(const std::string &str)
{
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";

  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

std::string Vocabulary::extract_value(const std::string &line)
{
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos)
    return "";

  std::string value = line.substr(colon_pos + 1);
  return trim(value);
}

std::string Vocabulary::extract_descriptor(const std::string &line)
{
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos)
    return "";

  std::string desc_part = line.substr(colon_pos + 1);
  desc_part = trim(desc_part);

  // 移除YAML的特殊字符，如引号
  if (!desc_part.empty() && desc_part.front() == '"')
    desc_part = desc_part.substr(1);
  if (!desc_part.empty() && desc_part.back() == '"')
    desc_part = desc_part.substr(0, desc_part.length() - 1);

  return desc_part;
}

/////  以上新增   ////////

bool Vocabulary::load(std::istream &ifile)
{
    uint64_t sig;//magic number describing the file
    ifile.read((char*)&sig,sizeof(sig));
    if (sig != 88877711233) // Check if it is a binary file.
        return false;

    ifile.seekg(0,std::ios::beg);
    fromStream(ifile);
    return true;
}


void Vocabulary::save(cv::FileStorage &f,
  const std::string &name) const
{

  f << name << "{";

  f << "k" << m_k;
  f << "L" << m_L;
  f << "scoringType" << m_scoring;
  f << "weightingType" << m_weighting;

  // tree
  f << "nodes" << "[";
  std::vector<NodeId> parents, children;
  std::vector<NodeId>::const_iterator pit;

  parents.push_back(0); // root

  while(!parents.empty())
  {
    NodeId pid = parents.back();
    parents.pop_back();

    const Node& parent = m_nodes[pid];
    children = parent.children;

    for(pit = children.begin(); pit != children.end(); pit++)
    {
      const Node& child = m_nodes[*pit];
      std::cout<<m_nodes[*pit].id<<" ";

      // save node data
      f << "{:";
      f << "nodeId" << (int)child.id;
      f << "parentId" << (int)pid;
      f << "weight" << (double)child.weight;
      f << "descriptor" << DescManip::toString(child.descriptor);
      f << "}";

      // add to parent list
      if(!child.isLeaf())
      {
        parents.push_back(*pit);
      }
    }
  }
  std::cout<<"\n";

  f << "]"; // nodes

  // words
  f << "words" << "[";

   for(auto wit = m_words.begin(); wit != m_words.end(); wit++)
  {
    WordId id = wit - m_words.begin();
    f << "{:";
    f << "wordId" << (int)id;
    f << "nodeId" << (int)(*wit)->id;
    f << "}";
  }

  f << "]"; // words

  f << "}";

}

void Vocabulary::toStream(  std::ostream &out_str, bool compressed) const throw(std::exception){

    uint64_t sig=88877711233;//magic number describing the file
    out_str.write((char*)&sig,sizeof(sig));
    out_str.write((char*)&compressed,sizeof(compressed));
    uint32_t nnodes=m_nodes.size();
    out_str.write((char*)&nnodes,sizeof(nnodes));
    if (nnodes==0)return;
    //save everything to a stream
    std::stringstream aux_stream;
    aux_stream.write((char*)&m_k,sizeof(m_k));
    aux_stream.write((char*)&m_L,sizeof(m_L));
    aux_stream.write((char*)&m_scoring,sizeof(m_scoring));
    aux_stream.write((char*)&m_weighting,sizeof(m_weighting));
    //nodes
    std::vector<NodeId> parents={0};// root


    while(!parents.empty())
    {
        NodeId pid = parents.back();
        parents.pop_back();

        const Node& parent = m_nodes[pid];

        for(auto pit :parent.children)
        {

            const Node& child = m_nodes[pit];
            aux_stream.write((char*)&child.id,sizeof(child.id));
            aux_stream.write((char*)&pid,sizeof(pid));
            aux_stream.write((char*)&child.weight,sizeof(child.weight));
            DescManip::toStream(child.descriptor,aux_stream);
            // add to parent list
            if(!child.isLeaf()) parents.push_back(pit);
        }
    }
    //words
    //save size
    uint32_t m_words_size=m_words.size();
    aux_stream.write((char*)&m_words_size,sizeof(m_words_size));
    for(auto wit = m_words.begin(); wit != m_words.end(); wit++)
    {
        WordId id = wit - m_words.begin();
        aux_stream.write((char*)&id,sizeof(id));
        aux_stream.write((char*)&(*wit)->id,sizeof((*wit)->id));
    }


    //now, decide if compress or not
    if (compressed){
        qlz_state_compress  state_compress;
        memset(&state_compress, 0, sizeof(qlz_state_compress));
        //Create output buffer
        int chunkSize=10000;
        std::vector<char> compressed( chunkSize+size_t(400), 0);
        std::vector<char> input( chunkSize, 0);
        int64_t total_size= static_cast<int64_t>(aux_stream.tellp());
        uint64_t total_compress_size=0;
        //calculate how many chunks will be written
        uint32_t nChunks= total_size / chunkSize;
        if ( total_size%chunkSize!=0) nChunks++;
        out_str.write((char*)&nChunks, sizeof(nChunks));
        //start compressing the chunks
		while (total_size != 0){
            int readSize=chunkSize;
            if (total_size<chunkSize) readSize=total_size;
            aux_stream.read(&input[0],readSize);
            uint64_t  compressed_size   = qlz_compress(&input[0], &compressed[0], readSize, &state_compress);
            total_size-=readSize;
            out_str.write(&compressed[0], compressed_size);
            total_compress_size+=compressed_size;
        }
    }
    else{
        out_str<<aux_stream.rdbuf();
    }
}


void Vocabulary:: load_fromtxt(const std::string &filename)throw(std::runtime_error){

    std::ifstream ifile(filename);
    if(!ifile)throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading:"+filename);
    int n1, n2;
    {
    std::string str;
    getline(ifile,str);
    std::stringstream ss(str);
    ss>>m_k>>m_L>>n1>>n2;
    }
    if(m_k<0 || m_k>20 || m_L<1 || m_L>10 || n1<0 || n1>5 || n2<0 || n2>3)
         throw std::runtime_error( "Vocabulary loading failure: This is not a correct text file!" );

    m_scoring = (ScoringType)n1;
    m_weighting = (WeightingType)n2;
    createScoringObject();
    // nodes
       int expected_nodes =
       (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
       m_nodes.reserve(expected_nodes);

       m_words.reserve(pow((double)m_k, (double)m_L + 1));

       m_nodes.resize(1);
       m_nodes[0].id = 0;

       int counter=0;
       while(!ifile.eof()){
           std::string snode;
           getline(ifile,snode);
           if (counter++%100==0)std::cerr<<".";
          // std::cout<<snode<<std::endl;
           if (snode.size()==0)break;
           std::stringstream ssnode(snode);

           int nid = m_nodes.size();
           m_nodes.resize(m_nodes.size()+1);
           m_nodes[nid].id = nid;

           int pid ;
           ssnode >> pid;
           m_nodes[nid].parent = pid;
           m_nodes[pid].children.push_back(nid);

           int nIsLeaf;
           ssnode >> nIsLeaf;

           //read until the end and add to data
           std::vector<float> data;data.reserve(100);
           float d;
           while( ssnode>>d) data.push_back(d);
           //the weight is the last
           m_nodes[nid].weight=data.back();
           data.pop_back();//remove
           //the rest, to the descriptor
           m_nodes[nid].descriptor.create(1,data.size(),CV_8UC1);
           auto ptr=m_nodes[nid].descriptor.ptr<uchar>(0);
           for(auto d:data) *ptr++=d;


           if(nIsLeaf>0)
           {
               int wid = m_words.size();
               m_words.resize(wid+1);

               m_nodes[nid].word_id = wid;
               m_words[wid] = &m_nodes[nid];
           }
           else
           {
               m_nodes[nid].children.reserve(m_k);
           }
       }
}
void Vocabulary::fromStream(  std::istream &str )   throw(std::exception){


    m_words.clear();
    m_nodes.clear();
    uint64_t sig=0;//magic number describing the file
    str.read((char*)&sig,sizeof(sig));
    if (sig!=88877711233) throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");
    bool compressed;
    str.read((char*)&compressed,sizeof(compressed));
    uint32_t nnodes;
    str.read((char*)&nnodes,sizeof(nnodes));
    if(nnodes==0)return;
    std::stringstream decompressed_stream;
    std::istream *_used_str=0;
    if (compressed){
        qlz_state_decompress state_decompress;
        memset(&state_decompress, 0, sizeof(qlz_state_decompress));
        int chunkSize=10000;
        std::vector<char> decompressed(chunkSize);
        std::vector<char> input(chunkSize+400);
        //read how many chunks are there
        uint32_t nChunks;
        str.read((char*)&nChunks,sizeof(nChunks));
        for(int i=0;i<nChunks;i++){
            str.read(&input[0],9);
            int c=qlz_size_compressed(&input[0]);
            str.read(&input[9],c-9);
            size_t d=qlz_decompress(&input[0], &decompressed[0], &state_decompress);
            decompressed_stream.write(&decompressed[0],d);
        }
        _used_str=&decompressed_stream;
    }
    else{
        _used_str=&str;
    }

    _used_str->read((char*)&m_k,sizeof(m_k));
    _used_str->read((char*)&m_L,sizeof(m_L));
    _used_str->read((char*)&m_scoring,sizeof(m_scoring));
    _used_str->read((char*)&m_weighting,sizeof(m_weighting));

    createScoringObject();
    m_nodes.resize(nnodes );
    m_nodes[0].id = 0;



    for(size_t i = 1; i < m_nodes.size(); ++i)
    {
        NodeId nid;
        _used_str->read((char*)&nid,sizeof(NodeId));
        Node& child = m_nodes[nid];
        child.id=nid;
        _used_str->read((char*)&child.parent,sizeof(child.parent));
        _used_str->read((char*)&child.weight,sizeof(child.weight));
        DescManip::fromStream(child.descriptor,*_used_str);
        m_nodes[child.parent].children.push_back(child.id);
     }
     //    // words
    uint32_t m_words_size;
    _used_str->read((char*)&m_words_size,sizeof(m_words_size));
    m_words.resize(m_words_size);
    for(unsigned int i = 0; i < m_words.size(); ++i)
    {
        WordId wid;NodeId nid;
        _used_str->read((char*)&wid,sizeof(wid));
        _used_str->read((char*)&nid,sizeof(nid));
        m_nodes[nid].word_id = wid;
        m_words[wid] = &m_nodes[nid];
    }
}
// --------------------------------------------------------------------------



void Vocabulary::load(const cv::FileStorage &fs,
  const std::string &name)
{
  m_words.clear();
  m_nodes.clear();

  cv::FileNode fvoc = fs[name];

  m_k = (int)fvoc["k"];
  m_L = (int)fvoc["L"];
  m_scoring = (ScoringType)((int)fvoc["scoringType"]);
  m_weighting = (WeightingType)((int)fvoc["weightingType"]);

  createScoringObject();

  // nodes
  cv::FileNode fn = fvoc["nodes"];

  m_nodes.resize(fn.size() + 1); // +1 to include root
  m_nodes[0].id = 0;

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId nid = (int)fn[i]["nodeId"];
    NodeId pid = (int)fn[i]["parentId"];
    WordValue weight = (WordValue)fn[i]["weight"];
    std::string d = (std::string)fn[i]["descriptor"];

    m_nodes[nid].id = nid;
    m_nodes[nid].parent = pid;
    m_nodes[nid].weight = weight;
    m_nodes[pid].children.push_back(nid);

    DescManip::fromString(m_nodes[nid].descriptor, d);
  }

  // words
  fn = fvoc["words"];

  m_words.resize(fn.size());

  for(unsigned int i = 0; i < fn.size(); ++i)
  {
    NodeId wid = (int)fn[i]["wordId"];
    NodeId nid = (int)fn[i]["nodeId"];

    m_nodes[nid].word_id = wid;
    m_words[wid] = &m_nodes[nid];
  }
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */

std::ostream& operator<<(std::ostream &os,
  const Vocabulary &voc)
{
  os << "Vocabulary: k = " << voc.getBranchingFactor()
    << ", L = " << voc.getDepthLevels()
    << ", Weighting = ";

  switch(voc.getWeightingType())
  {
    case TF_IDF: os << "tf-idf"; break;
    case TF: os << "tf"; break;
    case IDF: os << "idf"; break;
    case BINARY: os << "binary"; break;
  }

  os << ", Scoring = ";
  switch(voc.getScoringType())
  {
    case L1_NORM: os << "L1-norm"; break;
    case L2_NORM: os << "L2-norm"; break;
    case CHI_SQUARE: os << "Chi square distance"; break;
    case KL: os << "KL-divergence"; break;
    case BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
    case DOT_PRODUCT: os << "Dot product"; break;
  }

  os << ", Number of words = " << voc.size();

  return os;
}
/**
 * @brief Vocabulary::clear
 */
// void Vocabulary::clear(){
//     delete m_scoring_object;
//     m_scoring_object=0;
//     m_nodes.clear();
//     m_words.clear();

// }
int Vocabulary::getDescritorSize()const
{
    if (m_words.size()==0)return -1;
    else return m_words[0]->descriptor.cols;
}
int Vocabulary::getDescritorType()const{

    if (m_words.size()==0)return -1;
    else return m_words[0]->descriptor.type();
}


}
