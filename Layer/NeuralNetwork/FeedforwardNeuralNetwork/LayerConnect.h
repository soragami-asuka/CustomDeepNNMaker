//======================================
// レイヤー間の接続設定用クラス
//======================================
#ifndef __GRAVISBELL_LAYER_CONNECT_H__
#define __GRAVISBELL_LAYER_CONNECT_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>

#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>
#include<vector>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	static const S32 INVALID_DINPUTBUFFER_ID = 0xFFFF;
	static const S32 INVALID_OUTPUTBUFFER_ID = 0xFFFF;

	/** レイヤーのポインタと接続位置の情報 */
	struct LayerPosition
	{
		class ILayerConnect* pLayer;
		S32 position;

		LayerPosition()
			:	pLayer	(NULL)
			,	position(-1)
		{
		}
		LayerPosition(class ILayerConnect* pLayer)
			:	pLayer	(pLayer)
			,	position(-1)
		{
		}
	};

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
		virtual U32 GetLayerKind()const = 0;


		//====================================
		// 実行時設定
		//====================================
		/** 実行時設定を取得する. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter()const = 0;

		/** 実行時設定を設定する.
			int型、float型、enum型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		/** 実行時設定を設定する.
			int型、float型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		/** 実行時設定を設定する.
			bool型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		/** 実行時設定を設定する.
			string型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;



		//====================================
		// 入出力データ構造
		//====================================

		/** 出力データ構造を取得する.
			@return	出力データ構造 */
		virtual IODataStruct GetOutputDataStruct()const = 0;
		/** 出力データバッファを取得する.(ホストメモリ)
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer_d()const = 0;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		virtual S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const = 0;
		/** 入力誤差バッファを位置指定で取得する */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum_d(S32 num)const = 0;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		virtual ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const = 0;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		virtual ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList) = 0;

	public:
		/** レイヤーに入力レイヤーを追加する. */
		virtual ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer) = 0;
		/** レイヤーにバイパスレイヤーを追加する.*/
		virtual ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer) = 0;

		/** レイヤーから入力レイヤーを削除する */
		virtual ErrorCode EraseInputLayer(const Gravisbell::GUID& guid) = 0;
		/** レイヤーからバイパスレイヤーを削除する */
		virtual ErrorCode EraseBypassLayer(const Gravisbell::GUID& guid) = 0;

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetInputLayer() = 0;
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		virtual ErrorCode ResetBypassLayer() = 0;

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetInputLayerCount()const = 0;
		/** レイヤーに接続している入力レイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ILayerConnect* GetInputLayerByNum(U32 i_inputNum) = 0;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		virtual U32 GetBypassLayerCount()const = 0;
		/** レイヤーに接続しているバイパスレイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ILayerConnect* GetBypassLayerByNum(U32 i_inputNum) = 0;

		/** レイヤーの接続を解除 */
		virtual ErrorCode Disconnect(void) = 0;

		/** レイヤーで使用する出力バッファのIDを登録する */
		virtual ErrorCode SetOutputBufferID(S32 i_outputBufferID) = 0;

		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ErrorCode SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID) = 0;
		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual S32 GetDInputBufferID(U32 i_inputNum)const = 0;


	public:
		/** 出力先レイヤーを追加する */
		virtual ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer) = 0;
		/** 出力先レイヤーを削除する */
		virtual ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid) = 0;

		/** レイヤーに接続している出力先レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		virtual U32 GetOutputToLayerCount()const = 0;
		/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		virtual ILayerConnect* GetOutputToLayerByNum(U32 i_num) = 0;


	public:
		/** レイヤーの初期化処理.
			接続状況は維持したままレイヤーの中身を初期化する. */
		virtual ErrorCode Initialize(void) = 0;

		/** 接続の確立を行う */
		virtual ErrorCode EstablishmentConnection(void) = 0;

		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		virtual ErrorCode PreProcessLearn(unsigned int batchSize) = 0;
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcessCalculate(unsigned int batchSize) = 0;


		/** 処理ループの初期化処理.
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcessLoop() = 0;


		/** 演算処理を実行する. */
		virtual ErrorCode Calculate(void) = 0;
		/** 入力誤差計算を実行する. */
		virtual ErrorCode CalculateDInput(void) = 0;
		/** 学習処理を実行する. */
		virtual ErrorCode Training(void) = 0;
	};



}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_H__