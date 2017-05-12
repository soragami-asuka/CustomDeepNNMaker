//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>

#include"LayerConnect.h"
#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_Base : public INeuralNetwork
	{
	private:
		// データ本体
		class FeedforwardNeuralNetwork_LayerData_Base& layerData;

		std::map<Gravisbell::GUID, ILayerConnect*>	lpLayerInfo;	/**< 全レイヤーの管理クラス. <レイヤーGUID, レイヤー接続情報のアドレス> */

		std::list<ILayerConnect*> lpCalculateLayerList;		/**< レイヤーを処理順に並べたリスト.  */

		LayerConnectInput  inputLayer;	/**< 入力信号の代替レイヤーのアドレス. */
		LayerConnectOutput outputLayer;	/**< 出力信号の代替レイヤーのアドレス. */

		const Gravisbell::GUID guid;			/**< レイヤー識別用のGUID */

		SettingData::Standard::IData* pLearnData;		/**< 学習設定を定義したコンフィグクラス */

		U32 batchSize;	/**< バッチサイズ */

		// 演算時の入力データ
		CONST_BATCH_BUFFER_POINTER m_lppInputBuffer;	/**< 演算時の入力データ */
		CONST_BATCH_BUFFER_POINTER m_lppDOutputBuffer;	/**< 入力誤差計算時の出力誤差データ */

	public:
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
		/** コンストラクタ
			@param	i_inputGUID	入力信号に割り当てられたGUID.自分で作ることができないので外部で作成して引き渡す. */
		FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_Base();


		//====================================
		// レイヤーの追加
		//====================================
	public:
		/** レイヤーを追加する.
			追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param pLayer	追加するレイヤーのアドレス. */
		ErrorCode AddLayer(ILayerBase* pLayer);
		/** レイヤーを削除する.
			@param i_guid	削除するレイヤーのGUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** レイヤーを全削除する */
		ErrorCode EraseAllLayer();

		/** 登録されているレイヤー数を取得する */
		U32 GetLayerCount()const;
		/** レイヤーのGUIDを番号指定で取得する */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);


		//====================================
		// 入出力レイヤー
		//====================================
	public:
		/** 入力信号に割り当てられているGUIDを取得する */
		GUID GetInputGUID()const;

		/** 出力信号レイヤーを設定する */
		ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// レイヤーの接続
		//====================================
	public:
		/** レイヤーに入力レイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** レイヤーにバイパスレイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID);
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID);

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;


		/** レイヤーの接続状態に異常がないかチェックする.
			@param	o_errorLayer	エラーが発生したレイヤーGUID格納先. 
			@return	接続に異常がない場合はNO_ERROR, 異常があった場合は異常内容を返し、対象レイヤーのGUIDをo_errorLayerに格納する. */
		ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer);



		//====================================
		// 学習設定
		//====================================
	public:
		/** 学習設定を取得する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID. */
		const SettingData::Standard::IData* GetLearnSettingData(const Gravisbell::GUID& guid)const;
		SettingData::Standard::IData* GetLearnSettingData(const Gravisbell::GUID& guid);

		/** 学習設定のアイテムを取得する.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID. */
		SettingData::Standard::IItemBase* GetLearnSettingDataItem(const Gravisbell::GUID& guid, const wchar_t* i_dataID);

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const wchar_t* i_dataID, S32 i_param);
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const wchar_t* i_dataID, F32 i_param);
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const wchar_t* i_dataID, bool i_param);
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const wchar_t* i_dataID, const wchar_t* i_param);
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param);


		//====================================
		// 入出力バッファ関連
		//====================================
	public:
		/** 入力バッファを取得する */
		CONST_BATCH_BUFFER_POINTER GetInputBuffer()const;
		/** 出力差分バッファを取得する */
		CONST_BATCH_BUFFER_POINTER GetDOutputBuffer()const;
		
		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;

		/** 学習差分を取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
			@return	誤差差分配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		virtual ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;


		//===========================
		// レイヤー共通
		//===========================
	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const;

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤーの種類識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		U32 GetBatchSize()const;


		//================================
		// 初期化処理
		//================================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U32 GetUseBufferByteCount()const;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		U32 GetOutputBufferCount()const;


		//================================
		// 演算処理
		//================================
	protected:
		/** 接続の確立を行う */
		ErrorCode EstablishmentConnection(void);

	public:
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(U32 batchSize);
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);
		
		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
		/** 演算ループの初期化処理.データセットの演算開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculateLoop();

		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);


		//================================
		// 学習処理
		//================================
	public:
		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif	// __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_H__