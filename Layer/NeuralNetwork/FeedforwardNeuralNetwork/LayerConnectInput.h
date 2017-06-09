//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X
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


	/** ���C���[�̐ڑ��Ɋւ���N���X(���͐M���̑�p) */
	class LayerConnectInput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::vector<LayerPosition> lppOutputToLayer;	/**< �o�͐惌�C���[. SingleOutput�����Ȃ̂ŁA�K��1�� */

	public:
		/** �R���X�g���N�^ */
		LayerConnectInput(class FeedforwardNeuralNetwork_Base& neuralNetwork);
		/** �f�X�g���N�^ */
		virtual ~LayerConnectInput();

	public:
		/** GUID���擾���� */
		Gravisbell::GUID GetGUID()const;
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind()const;
		
		/** �w�K�ݒ�̃|�C���^���擾����.
			�擾�����f�[�^�𒼐ڏ��������邱�ƂŎ��̊w�K���[�v�ɔ��f����邪�ANULL���Ԃ��Ă��邱�Ƃ�����̂Œ���. */
		Gravisbell::SettingData::Standard::IData* GetLearnSettingData();

		/** �o�̓f�[�^�\�����擾����.
			@return	�o�̓f�[�^�\�� */
		IODataStruct GetOutputDataStruct()const;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;

		/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
		S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const;
		/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
		CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const;

		/** ���C���[���X�g���쐬����.
			@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
		ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const;
		/** �v�Z�������X�g���쐬����.
			@param	i_lpLayerGUID		�S���C���[��GUID.
			@param	io_lpCalculateList	���Z���ɕ��ׂ�ꂽ�ڑ����X�g.
			@param	io_lpAddedList		�ڑ����X�g�ɓo�^�ς݂̃��C���[��GUID�ꗗ.
			@param	io_lpAddWaitList	�ǉ��ҋ@��Ԃ̐ڑ��N���X�̃��X�g. */
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList);

	public:
		/** ���C���[�ɓ��̓��C���[��ǉ�����. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** ���C���[������̓��C���[���폜���� */
		ErrorCode EraseInputLayer(const Gravisbell::GUID& guid);
		/** ���C���[������̓��C���[���폜���� */
		ErrorCode EraseBypassLayer(const Gravisbell::GUID& guid);

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer();
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer();

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		U32 GetBypassLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

		/** ���C���[�̐ڑ������� */
		ErrorCode Disconnect(void);


		/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ErrorCode SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID);
		/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		S32 GetDInputBufferID(U32 i_inputNum)const;


		//==========================================
		// �o�̓��C���[�֘A
		//==========================================
	public:
		/** �o�͐惌�C���[��ǉ����� */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** �o�͐惌�C���[���폜���� */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);

		/** ���C���[�ɐڑ����Ă���o�͐惌�C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetOutputToLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetOutputToLayerByNum(U32 i_num);


	public:
		/** ���C���[�̏���������.
			�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
		ErrorCode Initialize(void);

		/** �ڑ��̊m�����s�� */
		ErrorCode EstablishmentConnection(void);

		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLearn(unsigned int batchSize);
		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);


		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLearnLoop();
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculateLoop();


		/** ���Z���������s����. */
		ErrorCode Calculate(void);
		/** �w�K���������s����. */
		ErrorCode Training(void);
	};

	
}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_INPUT_H__