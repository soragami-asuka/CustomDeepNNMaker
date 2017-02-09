//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include<INNLayer.h>

#include<vector>

using namespace CustomDeepNNLibrary;

typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

class NNLayer_FeedforwardBase : public CustomDeepNNLibrary::INNLayer
{
protected:
	GUID guid;

	INNLayerConfig* pConfig;

	std::vector<IOutputLayer*> lppInputFromLayer;		/**< 入力元レイヤーのリスト */
	std::vector<IInputLayer*>  lppOutputToLayer;	/**< 出力先レイヤーのリスト */

public:
	/** コンストラクタ */
	NNLayer_FeedforwardBase(GUID guid);

	/** デストラクタ */
	virtual ~NNLayer_FeedforwardBase();

	//===========================
	// レイヤー共通
	//===========================
public:
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	unsigned int GetUseBufferByteCount()const;

	/** レイヤー固有のGUIDを取得する */
	ELayerErrorCode GetGUID(GUID& o_guid)const;

	/** レイヤーの種類識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	ELayerErrorCode GetLayerCode(GUID& o_layerCode)const;

	/** 設定情報を設定 */
	ELayerErrorCode SetLayerConfig(const INNLayerConfig& config);
	/** レイヤーの設定情報を取得する */
	const INNLayerConfig* GetLayerConfig()const;


	//===========================
	// 入力レイヤー関連
	//===========================
public:
	/** 入力バッファ数を取得する. */
	unsigned int GetInputBufferCount()const;

public:
	/** 入力元レイヤーへのリンクを追加する.
		@param	pLayer	追加する入力元レイヤー
		@return	成功した場合0 */
	ELayerErrorCode AddInputFromLayer(IOutputLayer* pLayer);
	/** 入力元レイヤーへのリンクを削除する.
		@param	pLayer	削除する入力元レイヤー
		@return	成功した場合0 */
	ELayerErrorCode EraseInputFromLayer(IOutputLayer* pLayer);

public:
	/** 入力元レイヤー数を取得する */
	unsigned int GetInputFromLayerCount()const;
	/** 入力元レイヤーのアドレスを番号指定で取得する.
		@param num	取得するレイヤーの番号.
		@return	成功した場合入力元レイヤーのアドレス.失敗した場合はNULLが返る. */
	IOutputLayer* GetInputFromLayerByNum(unsigned int num)const;

	/** 入力元レイヤーが入力バッファのどの位置に居るかを返す.
		※対象入力レイヤーの前にいくつの入力バッファが存在するか.
		　学習差分の使用開始位置としても使用する.
		@return 失敗した場合負の値が返る*/
	int GetInputBufferPositionByLayer(const IOutputLayer* pLayer);


	//===========================
	// 出力レイヤー関連
	//===========================
public:
	/** 出力データ構造を取得する */
	IODataStruct GetOutputDataStruct()const;

	/** 出力バッファ数を取得する */
	unsigned int GetOutputBufferCount()const;

public:
	/** 出力先レイヤーへのリンクを追加する.
		@param	pLayer	追加する出力先レイヤー
		@return	成功した場合0 */
	ELayerErrorCode AddOutputToLayer(class IInputLayer* pLayer);
	/** 出力先レイヤーへのリンクを削除する.
		@param	pLayer	削除する出力先レイヤー
		@return	成功した場合0 */
	ELayerErrorCode EraseOutputToLayer(class IInputLayer* pLayer);

public:
	/** 出力先レイヤー数を取得する */
	unsigned int GetOutputToLayerCount()const;
	/** 出力先レイヤーのアドレスを番号指定で取得する.
		@param num	取得するレイヤーの番号.
		@return	成功した場合出力先レイヤーのアドレス.失敗した場合はNULLが返る. */
	IInputLayer* GetOutputToLayerByNum(unsigned int num)const;


	//===========================
	// 固有関数
	//===========================
public:
	/** ニューロン数を取得する */
	unsigned int GetNeuronCount()const;
};
