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


		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;

		/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
		virtual S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const = 0;
		/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum(S32 num)const = 0;

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

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetInputLayer() = 0;
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetBypassLayer() = 0;

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetInputLayerCount()const = 0;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ILayerConnect* GetInputLayerByNum(U32 i_inputNum) = 0;

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		virtual U32 GetBypassLayerCount()const = 0;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		virtual ILayerConnect* GetBypassLayerByNum(U32 i_inputNum) = 0;


	public:
		/** �o�͐惌�C���[��ǉ����� */
		virtual ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer) = 0;
		/** �o�͐惌�C���[���폜���� */
		virtual ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid) = 0;

	public:
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


		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data) = 0;
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessCalculateLoop() = 0;


		/** ���Z���������s����. */
		virtual ErrorCode Calculate(void) = 0;
		/** �w�K�덷���v�Z����. */
		virtual ErrorCode CalculateLearnError(void) = 0;
		/** �w�K���������C���[�ɔ��f������.*/
		virtual ErrorCode ReflectionLearnError(void) = 0;
	};


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

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer();
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer();

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount()const;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		U32 GetBypassLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);


	protected:
		/** �o�͐惌�C���[��ǉ����� */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** �o�͐惌�C���[���폜���� */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);

		
	public:
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
		ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculateLoop();


		/** ���Z���������s����. */
		ErrorCode Calculate(void);
		/** �w�K�덷���v�Z����. */
		ErrorCode CalculateLearnError(void);
		/** �w�K���������C���[�ɔ��f������.*/
		ErrorCode ReflectionLearnError(void);
	};

	/** ���C���[�̐ڑ��Ɋւ���N���X(�o�͐M���̑�p) */
	class LayerConnectOutput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::vector<ILayerConnect*> lppInputFromLayer;	/**< ���͌����C���[. SingleInput�����Ȃ̂ŁA�K��1�� */

	public:
		/** �R���X�g���N�^ */
		LayerConnectOutput(class FeedforwardNeuralNetwork_Base& neuralNetwork);
		/** �f�X�g���N�^ */
		virtual ~LayerConnectOutput();

	public:
		/** GUID���擾���� */
		Gravisbell::GUID GetGUID()const;
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind()const;

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

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer();
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer();

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount()const;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		U32 GetBypassLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

	protected:
		/** �o�͐惌�C���[��ǉ����� */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** �o�͐惌�C���[���폜���� */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);


	public:
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
		ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculateLoop();


		/** ���Z���������s����. */
		ErrorCode Calculate(void);
		/** �w�K�덷���v�Z����. */
		ErrorCode CalculateLearnError(void);
		/** �w�K���������C���[�ɔ��f������.*/
		ErrorCode ReflectionLearnError(void);
	};


	/** ���C���[�̐ڑ��Ɋւ���N���X(�P�����,�P��o��) */
	class LayerConnectSingle2Single : public ILayerConnect
	{
	public:
		INNLayer* pLayer;	/**< ���C���[�̃A�h���X */

		std::vector<LayerPosition>  lppOutputToLayer;		/**< �o�͐惌�C���[. SingleOutput�����Ȃ̂ŁA�K��1�� */
		std::vector<ILayerConnect*> lppInputFromLayer;	/**< ���͌����C���[. SingleInput�����Ȃ̂ŁA�K��1�� */

	public:
		/** �R���X�g���N�^ */
		LayerConnectSingle2Single(class INNLayer* pLayer);
		/** �f�X�g���N�^ */
		virtual ~LayerConnectSingle2Single();

	public:
		/** GUID���擾���� */
		Gravisbell::GUID GetGUID()const;
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind()const;

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

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer();
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer();

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount()const;
		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
		U32 GetBypassLayerCount()const;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

	protected:
		/** �o�͐惌�C���[��ǉ����� */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** �o�͐惌�C���[���폜���� */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);


	public:
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
		ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculateLoop();


		/** ���Z���������s����. */
		ErrorCode Calculate(void);
		/** �w�K�덷���v�Z����. */
		ErrorCode CalculateLearnError(void);
		/** �w�K���������C���[�ɔ��f������.*/
		ErrorCode ReflectionLearnError(void);
	};

}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_H__