//======================================
// レイヤー間の接続設定用クラス
//======================================
#ifndef __GRAVISBELL_LAYER_CONNECT_INPUT_H__
#define __GRAVISBELL_LAYER_CONNECT_INPUT_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>

#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>
#include<vector>

#include"LayerConnect.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** レイヤーの接続に関するクラス(入力信号の代用) */
	class LayerConnectInput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::vector<LayerPosition> lppOutputToLayer;	/**< 出力先レイヤー. SingleOutput扱いなので、必ず1個 */

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
		U32 GetLayerKind()const;
		
		/** 学習設定のポインタを取得する.
			取得したデータを直接書き換えることで次の学習ループに反映されるが、NULLが返ってくることもあるので注意. */
		Gravisbell::SettingData::Standard::IData* GetLearnSettingData();

		/** 出力データ構造を取得する.
			@return	出力データ構造 */
		IODataStruct GetOutputDataStruct()const;
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
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList);

	public:
		/** レイヤーに入力レイヤーを追加する. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** レイヤーにバイパスレイヤーを追加する.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** レイヤーから入力レイヤーを削除する */
		ErrorCode EraseInputLayer(const Gravisbell::GUID& guid);
		/** レイヤーから入力レイヤーを削除する */
		ErrorCode EraseBypassLayer(const Gravisbell::GUID& guid);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer();
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer();

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

		/** レイヤーの接続を解除 */
		ErrorCode Disconnect(void);


		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ErrorCode SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID);
		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		S32 GetDInputBufferID(U32 i_inputNum)const;


		//==========================================
		// 出力レイヤー関連
		//==========================================
	public:
		/** 出力先レイヤーを追加する */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** 出力先レイヤーを削除する */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);

		/** レイヤーに接続している出力先レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetOutputToLayerCount()const;
		/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetOutputToLayerByNum(U32 i_num);


	public:
		/** レイヤーの初期化処理.
			接続状況は維持したままレイヤーの中身を初期化する. */
		ErrorCode Initialize(void);

		/** 接続の確立を行う */
		ErrorCode EstablishmentConnection(void);

		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(unsigned int batchSize);
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);


		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessLearnLoop();
		/** 演算ループの初期化処理.データセットの演算開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculateLoop();


		/** 演算処理を実行する. */
		ErrorCode Calculate(void);
		/** 学習処理を実行する. */
		ErrorCode Training(void);
	};

	
}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_INPUT_H__