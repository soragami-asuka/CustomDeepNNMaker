//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X
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

	/** ���C���[�̃|�C���^�Ɛڑ��ʒu�̏�� */
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

	/** ���C���[�̐ڑ��Ɋւ���N���X */
	class ILayerConnect
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConnect(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerConnect(){}

	public:
		/** GUID���擾���� */
		virtual Gravisbell::GUID GetGUID()const = 0;
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		virtual U32 GetLayerKind()const = 0;


		//====================================
		// ���s���ݒ�
		//====================================
		/** ���s���ݒ���擾����. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter()const = 0;

		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			bool�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			string�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;



		//====================================
		// ���o�̓f�[�^�\��
		//====================================

		/** �o�̓f�[�^�\�����擾����.
			@return	�o�̓f�[�^�\�� */
		virtual IODataStruct GetOutputDataStruct()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.(�z�X�g������)
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer_d()const = 0;

		/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
		virtual S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const = 0;
		/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum_d(S32 num)const = 0;

		/** ���C���[���X�g���쐬����.
			@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
		virtual ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const = 0;
		/** �v�Z�������X�g���쐬����.
			@param	i_lpLayerGUID		�S���C���[��GUID.
			@param	io_lpCalculateList	���Z���ɕ��ׂ�ꂽ�ڑ����X�g.
			@param	io_lpAddedList		�ڑ����X�g�ɓo�^�ς݂̃��C���[��GUID�ꗗ.
			@param	io_lpAddWaitList	�ǉ��ҋ@��Ԃ̐ڑ��N���X�̃��X�g. */
		virtual ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList) = 0;

	public:
		/** ���C���[�ɓ��̓��C���[��ǉ�����. */
		virtual ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer) = 0;
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
		virtual ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer) = 0;

		/** ���C���[������̓��C���[���폜���� */
		virtual ErrorCode EraseInputLayer(const Gravisbell::GUID& guid) = 0;
		/** ���C���[����o�C�p�X���C���[���폜���� */
		virtual ErrorCode EraseBypassLayer(const Gravisbell::GUID& guid) = 0;

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetInputLayer() = 0;
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetBypassLayer() = 0;

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetInputLayerCount()const = 0;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ILayerConnect* GetInputLayerByNum(U32 i_inputNum) = 0;

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		virtual U32 GetBypassLayerCount()const = 0;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ILayerConnect* GetBypassLayerByNum(U32 i_inputNum) = 0;

		/** ���C���[�̐ڑ������� */
		virtual ErrorCode Disconnect(void) = 0;

		/** ���C���[�Ŏg�p����o�̓o�b�t�@��ID��o�^���� */
		virtual ErrorCode SetOutputBufferID(S32 i_outputBufferID) = 0;

		/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ErrorCode SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID) = 0;
		/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual S32 GetDInputBufferID(U32 i_inputNum)const = 0;


	public:
		/** �o�͐惌�C���[��ǉ����� */
		virtual ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer) = 0;
		/** �o�͐惌�C���[���폜���� */
		virtual ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid) = 0;

		/** ���C���[�ɐڑ����Ă���o�͐惌�C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetOutputToLayerCount()const = 0;
		/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ILayerConnect* GetOutputToLayerByNum(U32 i_num) = 0;


	public:
		/** ���C���[�̏���������.
			�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
		virtual ErrorCode Initialize(void) = 0;

		/** �ڑ��̊m�����s�� */
		virtual ErrorCode EstablishmentConnection(void) = 0;

		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLearn(unsigned int batchSize) = 0;
		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessCalculate(unsigned int batchSize) = 0;


		/** �������[�v�̏���������.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLoop() = 0;


		/** ���Z���������s����. */
		virtual ErrorCode Calculate(void) = 0;
		/** ���͌덷�v�Z�����s����. */
		virtual ErrorCode CalculateDInput(void) = 0;
		/** �w�K���������s����. */
		virtual ErrorCode Training(void) = 0;
	};



}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_H__