//======================================
// レイヤー間の接続設定用クラス
//======================================
#include<Layer/NeuralNetwork/INeuralNetwork.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>



namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** レイヤーの接続に関するクラス */
	class ILayerConnect
	{
	public:
		/** コンストラクタ */
		ILayerConnect(){}
		/** デストラクタ */
		virtual ~ILayerConnect(){}

	public:
		/** GUIDを取得する */
		virtual Gravisbell::GUID GetGUID()const = 0;
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		virtual unsigned int GetLayerKind()const = 0;


		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		virtual S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const = 0;
		/** 入力誤差バッファを位置指定で取得する */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const = 0;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		virtual ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const = 0;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		virtual ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)const = 0;

		/** 演算事前処理.
			接続の確立を行う. */
		virtual ErrorCode PreCalculate(void) = 0;

		/** 演算処理を実行する. */
		virtual ErrorCode Calculate(void) = 0;
		/** 学習誤差を計算する. */
		virtual ErrorCode CalculateLearnError(void) = 0;
		/** 学習差分をレイヤーに反映させる.*/
		virtual ErrorCode ReflectionLearnError(void) = 0;

	public:
		/** レイヤーに入力レイヤーを追加する. */
		virtual ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer) = 0;
		/** レイヤーにバイパスレイヤーを追加する.*/
		virtual ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer) = 0;

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetInputLayer() = 0;
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetBypassLayer() = 0;

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetInputLayerCount()const = 0;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ILayerConnect* GetInputLayerByNum(U32 i_inputNum) = 0;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		virtual U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ILayerConnect* GetBypassLayerByNum(U32 i_inputNum) = 0;


	protected:
		/** 出力先レイヤーを追加する */
		virtual ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer) = 0;
		/** 出力先レイヤーを削除する */
		virtual ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid) = 0;
	};


	/** レイヤーの接続に関するクラス(入力信号の代用) */
	class LayerConnectInput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::list<ILayerConnect*> lppOutputToLayer;	/**< 出力先レイヤー. SingleOutput扱いなので、必ず1個 */

	public:
		/** コンストラクタ */
		LayerConnectInput(class FeedforwardNeuralNetwork_Base& neuralNetwork);
		/** デストラクタ */
		virtual ~LayerConnectInput();

	public:
		/** GUIDを取得する */
		Gravisbell::GUID GetGUID()const;
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		unsigned int GetLayerKind()const;

		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const;
		/** 入力誤差バッファを位置指定で取得する */
		CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)const;

		/** 演算事前処理.
			接続の確立を行う. */
		ErrorCode PreCalculate(void);

		/** 演算処理を実行する. */
		ErrorCode Calculate(void);
		/** 学習誤差を計算する. */
		ErrorCode CalculateLearnError(void);
		/** 学習差分をレイヤーに反映させる.*/
		ErrorCode ReflectionLearnError(void);

	public:
		/** レイヤーに入力レイヤーを追加する. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** レイヤーにバイパスレイヤーを追加する.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer();
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer();

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount()const;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);


	protected:
		/** 出力先レイヤーを追加する */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** 出力先レイヤーを削除する */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);
	};

	/** レイヤーの接続に関するクラス(出力信号の代用) */
	class LayerConnectOutput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::list<ILayerConnect*> lppInputFromLayer;	/**< 入力元レイヤー. SingleInput扱いなので、必ず1個 */

	public:
		/** コンストラクタ */
		LayerConnectOutput(class FeedforwardNeuralNetwork_Base& neuralNetwork);
		/** デストラクタ */
		virtual ~LayerConnectOutput();

	public:
		/** GUIDを取得する */
		Gravisbell::GUID GetGUID()const;
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		unsigned int GetLayerKind()const;

		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const;
		/** 入力誤差バッファを位置指定で取得する */
		CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)const;

		/** 演算事前処理.
			接続の確立を行う. */
		ErrorCode PreCalculate(void);

		/** 演算処理を実行する. */
		ErrorCode Calculate(void);
		/** 学習誤差を計算する. */
		ErrorCode CalculateLearnError(void);
		/** 学習差分をレイヤーに反映させる.*/
		ErrorCode ReflectionLearnError(void);

	public:
		/** レイヤーに入力レイヤーを追加する. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** レイヤーにバイパスレイヤーを追加する.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer();
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer();

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount()const;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

	protected:
		/** 出力先レイヤーを追加する */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** 出力先レイヤーを削除する */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);
	};


	/** レイヤーの接続に関するクラス(単一入力,単一出力) */
	class LayerConnectSingle2Single : public ILayerConnect
	{
	public:
		INNLayer* pLayer;	/**< レイヤーのアドレス */
		
		std::list<ILayerConnect*> lppOutputToLayer;		/**< 出力先レイヤー. SingleOutput扱いなので、必ず1個 */
		std::list<ILayerConnect*> lppInputFromLayer;	/**< 入力元レイヤー. SingleInput扱いなので、必ず1個 */

	public:
		/** コンストラクタ */
		LayerConnectSingle2Single(class INNLayer* pLayer);
		/** デストラクタ */
		virtual ~LayerConnectSingle2Single();

	public:
		/** GUIDを取得する */
		Gravisbell::GUID GetGUID()const;
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		unsigned int GetLayerKind()const;

		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const;
		/** 入力誤差バッファを位置指定で取得する */
		CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)const;

		/** 演算事前処理.
			接続の確立を行う. */
		ErrorCode PreCalculate(void);

		/** 演算処理を実行する. */
		ErrorCode Calculate(void);
		/** 学習誤差を計算する. */
		ErrorCode CalculateLearnError(void);
		/** 学習差分をレイヤーに反映させる.*/
		ErrorCode ReflectionLearnError(void);

	public:
		/** レイヤーに入力レイヤーを追加する. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** レイヤーにバイパスレイヤーを追加する.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer();
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer();

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount()const;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

	protected:
		/** 出力先レイヤーを追加する */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** 出力先レイヤーを削除する */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);
	};

}	// Gravisbell
}	// Layer
}	// NeuralNetwork